#!/usr/bin/env python3
"""
spaCy NER Domain Adaptation Pipeline
======================================
Converts cleaned CoNLL BIO files to spaCy format and trains.

PREREQUISITES:
    pip install spacy
    python -m spacy download nl_core_news_lg

USAGE:
    # Step 1: Convert CoNLL → .spacy (must do this first)
    python3 train_ner.py convert \
        --train training_data_v2_clean.conll \
        --dev dev_data_v2_clean.conll \
        --output-dir ./spacy_data

    # Step 2: Train (uses converted data)
    python3 train_ner.py train \
        --data-dir ./spacy_data \
        --output-dir ./ner_model \
        --config ner_config.cfg

    # Or do both in sequence:
    python3 train_ner.py all \
        --train training_data_v2_clean.conll \
        --dev dev_data_v2_clean.conll \
        --output-dir ./ner_model
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Step 1: Convert CoNLL → spaCy DocBin
# =========================================================================

def convert_conll_to_spacy(conll_path: Path, output_path: Path, n_sents: int = 1):
    """
    Stream-convert a CoNLL BIO file to spaCy .spacy (DocBin) format.
    Processes one sentence at a time to handle large files.
    """
    from spacy.lang.nl import Dutch
    from spacy.tokens import DocBin, Doc

    nlp = Dutch()
    db = DocBin()

    total_docs = 0
    total_ents = 0
    total_sents = 0
    skipped = 0

    sentence_buffer = []  # list of (tokens, tags)

    def flush_buffer():
        nonlocal total_docs, total_ents, skipped
        if not sentence_buffer:
            return

        all_tokens = []
        all_tags = []
        for sent_tokens, sent_tags in sentence_buffer:
            all_tokens.extend(sent_tokens)
            all_tags.extend(sent_tags)

        doc = Doc(nlp.vocab, words=all_tokens)

        # Convert BIO tags to entity spans
        entities = []
        i = 0
        while i < len(all_tags):
            if all_tags[i] == 'B-PER':
                start = i
                i += 1
                while i < len(all_tags) and all_tags[i] == 'I-PER':
                    i += 1
                span = doc.char_span(
                    doc[start].idx,
                    doc[i - 1].idx + len(doc[i - 1].text),
                    label='PER'
                )
                if span is not None:
                    entities.append(span)
            else:
                i += 1

        try:
            doc.ents = entities
            total_ents += len(doc.ents)
            db.add(doc)
            total_docs += 1
        except ValueError:
            skipped += 1

    current_tokens = []
    current_tags = []

    log.info("Converting %s → %s ...", conll_path, output_path)
    t0 = time.time()

    with open(conll_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentence_buffer.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                    total_sents += 1

                    if len(sentence_buffer) >= n_sents:
                        flush_buffer()
                        sentence_buffer = []

                    if total_sents % 50000 == 0:
                        log.info("  %d sentences → %d docs, %d entities",
                                 total_sents, total_docs, total_ents)
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[1])

    # Flush remaining
    if current_tokens:
        sentence_buffer.append((current_tokens, current_tags))
    if sentence_buffer:
        flush_buffer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(output_path)

    elapsed = time.time() - t0
    log.info("  DONE: %d docs, %d entities, %d skipped (%.0fs)",
             total_docs, total_ents, skipped, elapsed)
    return total_docs, total_ents


# =========================================================================
# Step 2: Write config file
# =========================================================================

def write_config(output_path: Path):
    """Write the spaCy training config for domain adaptation."""
    config_text = """[paths]
train = null
dev = null

[system]
gpu_allocator = null
seed = 42

[nlp]
lang = "nl"
pipeline = ["tok2vec", "ner"]
batch_size = 1000
disabled = []

[components]

[components.tok2vec]
source = "nl_core_news_lg"

[components.ner]
source = "nl_core_news_lg"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 500
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 500
frozen_components = ["tok2vec"]
annotating_components = []
before_to_disk = null
before_update = null

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 1e-08
learn_rate = 0.001

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = true

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.tokenizer]

[initialize.components]

[pretraining]

[nlp.tokenizer]
@tokenizers = "spacy.Tokenizer.v1"
"""
    output_path.write_text(config_text, encoding='utf-8')
    log.info("Config written to %s", output_path)


# =========================================================================
# Step 3: Train
# =========================================================================

def train_model(data_dir: Path, output_dir: Path, config_path: Path):
    """Run spacy train."""
    import subprocess

    train_path = data_dir / "train.spacy"
    dev_path = data_dir / "dev.spacy"

    if not train_path.exists():
        log.error("Train data not found: %s", train_path)
        log.error("Run 'python3 train_ner.py convert' first.")
        sys.exit(1)
    if not dev_path.exists():
        log.error("Dev data not found: %s", dev_path)
        log.error("Run 'python3 train_ner.py convert' first.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(train_path),
        "--paths.dev", str(dev_path),
    ]

    log.info("Running: %s", " ".join(cmd))
    log.info("")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Training failed with return code %d", result.returncode)
        sys.exit(1)

    # Check outputs
    best_model = output_dir / "model-best"
    last_model = output_dir / "model-last"
    if best_model.exists():
        log.info("")
        log.info("Training complete!")
        log.info("  Best model: %s", best_model)
        log.info("  Last model: %s", last_model)
        log.info("")
        log.info("To evaluate:")
        log.info("  python -m spacy evaluate %s ./spacy_data/dev.spacy", best_model)
        log.info("")
        log.info("To use in your NER pipeline:")
        log.info("  nlp = spacy.load('%s')", best_model)
    else:
        log.error("Expected model output not found at %s", best_model)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="spaCy NER domain adaptation pipeline"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Convert command
    p_convert = subparsers.add_parser('convert', help='Convert CoNLL → .spacy')
    p_convert.add_argument('--train', required=True, help='Train CoNLL file')
    p_convert.add_argument('--dev', required=True, help='Dev CoNLL file')
    p_convert.add_argument('--output-dir', default='./spacy_data',
                           help='Output directory for .spacy files')

    # Train command
    p_train = subparsers.add_parser('train', help='Train the model')
    p_train.add_argument('--data-dir', default='./spacy_data',
                         help='Directory containing train.spacy and dev.spacy')
    p_train.add_argument('--output-dir', default='./ner_model',
                         help='Output directory for trained model')
    p_train.add_argument('--config', default=None,
                         help='Config file (auto-generated if not provided)')

    # All-in-one command
    p_all = subparsers.add_parser('all', help='Convert + train in sequence')
    p_all.add_argument('--train', required=True, help='Train CoNLL file')
    p_all.add_argument('--dev', required=True, help='Dev CoNLL file')
    p_all.add_argument('--output-dir', default='./ner_model',
                       help='Output directory for trained model')
    p_all.add_argument('--config', default=None,
                        help='Config file (auto-generated if not provided)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'convert':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_conll_to_spacy(
            Path(args.train),
            output_dir / "train.spacy"
        )
        convert_conll_to_spacy(
            Path(args.dev),
            output_dir / "dev.spacy"
        )

        log.info("")
        log.info("Conversion complete. Next step:")
        log.info("  python3 train_ner.py train --data-dir %s", output_dir)

    elif args.command == 'train':
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)

        if args.config:
            config_path = Path(args.config)
        else:
            config_path = data_dir / "ner_config.cfg"
            if not config_path.exists():
                write_config(config_path)

        train_model(data_dir, output_dir, config_path)

    elif args.command == 'all':
        data_dir = Path(args.output_dir) / "spacy_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)

        # Convert
        convert_conll_to_spacy(Path(args.train), data_dir / "train.spacy")
        convert_conll_to_spacy(Path(args.dev), data_dir / "dev.spacy")

        # Write config
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = data_dir / "ner_config.cfg"
            write_config(config_path)

        # Train
        train_model(data_dir, output_dir, config_path)


if __name__ == "__main__":
    main()
