#!/usr/bin/env python3
"""
Post-process CoNLL training data for spaCy NER domain adaptation.
Streaming version — processes one sentence at a time.

Fixes from quality audit:
  A. Strip trailing punctuation from entity-final tokens (7.8%)
  B. Strip trailing non-name function words (1.5%)
  C. Strip trailing particles on short (≤3 tok) entities (1.0%)
  D. Remove detectable false positives (0.2%)
  E. Chunk long sentences to max ~250 tokens for spaCy training

Usage:
    python3 clean_training_data.py --input training_data.conll --output training_data_clean.conll
    python3 clean_training_data.py --input training_data.conll --dry-run
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FUNCTION_WORDS = {
    'en', 'ende', 'als', 'door', 'voor', 'met', 'die', 'dat', 'het',
    'etc', 'fol', 'aan', 'bij', 'tot', 'uit', 'naar',
    'over', 'onder', 'hun', 'haar', 'zijn', 'mijn', 'ook', 'nog', 'wel',
    'niet', 'zeer', 'heeren', 'heer', 'vrouw', 'vrouwe', 'off', 'ofte',
    'om', 'op', 'so', 'soo', 'dan', 'maar', 'maer', 'int', 'ind',
}

TRUNCATION_PARTICLES = {'de', 'van', 'der', 'den', 'ter', 'ten', 'des',
                        'vander', 'vande', 'von'}

PARTICLE_STRIP_MAX_LEN = 3
STRIP_CHARS = '.,;:!?)]\u200b'


def is_false_positive(tokens):
    if not tokens:
        return True
    text = ' '.join(tokens).lower()
    if text in ('des heeren', 'den heere', 'den here', 'des heren',
                'den heer', 'onse heere', 'onsen heere'):
        return True
    if text.startswith('alle '):
        return True
    if tokens[0][0].isdigit():
        return True
    if len(tokens) == 1 and len(tokens[0].rstrip(STRIP_CHARS)) <= 2:
        return True
    return False


def clean_sentence(pairs):
    """
    Clean one sentence (list of (token, tag) pairs).
    Returns (new_pairs, stats).
    
    Strategy: extract entities, clean them (modifying both tokens and tags),
    then rebuild the full sentence with overrides.
    """
    stats = dict(punct=0, func=0, particle=0, fp=0, removed=0)

    # Extract entity spans
    entities = []  # (start, end_inclusive)
    i = 0
    while i < len(pairs):
        if pairs[i][1] == 'B-PER':
            start = i
            i += 1
            while i < len(pairs) and pairs[i][1] == 'I-PER':
                i += 1
            entities.append((start, i - 1))
        else:
            i += 1

    # Build override maps
    tag_override = {}   # position → new tag
    token_override = {} # position → new token text

    for start, end in entities:
        ent_tokens = [pairs[j][0] for j in range(start, end + 1)]
        orig_len = len(ent_tokens)

        # A: Strip trailing punctuation from last token's TEXT
        if ent_tokens:
            last = ent_tokens[-1]
            stripped = last.rstrip(STRIP_CHARS)
            if stripped != last:
                stats['punct'] += 1
                if stripped:
                    ent_tokens[-1] = stripped
                else:
                    # Entire token was punct — drop it
                    ent_tokens = ent_tokens[:-1]

        # Also strip leading punct/junk from first token
        if ent_tokens:
            first = ent_tokens[0]
            lstripped = first.lstrip('(["\'')
            if lstripped != first:
                if lstripped:
                    ent_tokens[0] = lstripped
                else:
                    ent_tokens = ent_tokens[1:]

        # B: Strip trailing function words (iterative)
        while len(ent_tokens) >= 2:
            lw = ent_tokens[-1].lower().rstrip(STRIP_CHARS)
            if lw in FUNCTION_WORDS:
                ent_tokens = ent_tokens[:-1]
                stats['func'] += 1
            else:
                break

        # C: Strip truncated particles on short entities
        if len(ent_tokens) <= PARTICLE_STRIP_MAX_LEN:
            modified = False
            while len(ent_tokens) >= 2:
                lw = ent_tokens[-1].lower().rstrip(STRIP_CHARS)
                if lw in TRUNCATION_PARTICLES:
                    ent_tokens = ent_tokens[:-1]
                    modified = True
                else:
                    break
            if modified:
                stats['particle'] += 1

        # A2: Re-strip trailing punct (may have been exposed by B/C)
        if ent_tokens:
            last = ent_tokens[-1]
            stripped = last.rstrip(STRIP_CHARS)
            if stripped != last:
                if stripped:
                    ent_tokens[-1] = stripped
                else:
                    ent_tokens = ent_tokens[:-1]

        # D: False positive check
        if is_false_positive(ent_tokens):
            stats['fp'] += 1
            for pos in range(start, end + 1):
                tag_override[pos] = 'O'
            continue

        new_len = len(ent_tokens)

        if new_len == 0:
            stats['removed'] += 1
            for pos in range(start, end + 1):
                tag_override[pos] = 'O'
        else:
            # Update token text for kept positions
            for k in range(new_len):
                pos = start + k
                token_override[pos] = ent_tokens[k]
            # Tag trimmed tail as O
            for pos in range(start + new_len, end + 1):
                tag_override[pos] = 'O'

    # Rebuild sentence
    result = []
    for i, (token, tag) in enumerate(pairs):
        new_token = token_override.get(i, token)
        new_tag = tag_override.get(i, tag)
        result.append((new_token, new_tag))

    return result, stats


def chunk_sentence(pairs, max_tokens):
    if len(pairs) <= max_tokens:
        return [pairs]
    chunks = []
    current = []
    for token, tag in pairs:
        if len(current) >= max_tokens and tag != 'I-PER':
            if current:
                chunks.append(current)
                current = []
        current.append((token, tag))
    if current:
        chunks.append(current)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + '_clean' + input_path.suffix)

    total = dict(punct=0, func=0, particle=0, fp=0, removed=0)
    in_sents = 0
    out_sents = 0
    in_entities = 0
    out_entities = 0
    out_tokens = 0
    bio_errors = 0
    over_max = 0
    out_file = None

    if not args.dry_run:
        out_file = open(output_path, 'w', encoding='utf-8')

    def process(sent):
        nonlocal in_sents, out_sents, in_entities, out_entities
        nonlocal out_tokens, bio_errors, over_max

        in_sents += 1
        in_entities += sum(1 for _, tag in sent if tag == 'B-PER')

        cleaned, stats = clean_sentence(sent)
        for k in total:
            total[k] += stats[k]

        for chunk in chunk_sentence(cleaned, args.max_tokens):
            out_sents += 1
            out_tokens += len(chunk)
            out_entities += sum(1 for _, tag in chunk if tag == 'B-PER')
            if len(chunk) > args.max_tokens:
                over_max += 1
            prev = 'O'
            for _, tag in chunk:
                if tag == 'I-PER' and prev == 'O':
                    bio_errors += 1
                prev = tag
            if out_file:
                for token, tag in chunk:
                    out_file.write(f"{token}\t{tag}\n")
                out_file.write("\n")

        if in_sents % 10000 == 0:
            log.info("  processed %d sentences...", in_sents)

    log.info("Reading %s...", input_path)
    current = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    process(current)
                    current = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current.append((parts[0], parts[1]))
    if current:
        process(current)

    if out_file:
        out_file.close()

    # Count PER density from output
    per_tokens = 0
    if not args.dry_run and output_path.exists():
        with open(output_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2 and parts[1] != 'O':
                        per_tokens += 1

    log.info("")
    log.info("=" * 60)
    log.info("CLEANING REPORT")
    log.info("=" * 60)
    log.info("Trailing punctuation fixes:     %6d", total['punct'])
    log.info("Trailing function word strips:   %6d", total['func'])
    log.info("Truncated particle strips:       %6d", total['particle'])
    log.info("False positives removed:         %6d", total['fp'])
    log.info("Entities collapsed to empty:     %6d", total['removed'])
    log.info("")
    log.info("Entities: %d → %d  (removed %d, %.1f%%)",
             in_entities, out_entities,
             in_entities - out_entities,
             (in_entities - out_entities) / max(in_entities, 1) * 100)
    log.info("Sentences: %d → %d  (chunking created %d new)",
             in_sents, out_sents, out_sents - in_sents)
    log.info("Tokens: %d", out_tokens)
    if per_tokens:
        log.info("PER token density: %.2f%%", per_tokens / max(out_tokens, 1) * 100)
    log.info("")
    log.info("Post-chunk stats:")
    log.info("  Still > %d tokens: %d", args.max_tokens, over_max)
    log.info("  BIO errors: %d", bio_errors)

    if not args.dry_run:
        size_mb = output_path.stat().st_size / 1e6
        log.info("Output: %s (%.1f MB)", output_path, size_mb)
    else:
        log.info("[DRY RUN — no output written]")


if __name__ == "__main__":
    main()
