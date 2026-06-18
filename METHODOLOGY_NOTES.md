# Methodology notes

Supplementary detail on methodological decisions and parameter choices that the thesis text could not accommodate. Section numbers refer to the thesis chapter structure. All parameter values are recoverable from the scripts themselves; this document explains why those values were chosen.

## §3.3.3 HTR collection infrastructure

HTR text was fetched from two machines (macOS, Windows) running `fetch_htr_corpus_fast.py` concurrently against the same Transkribus Read API. Collection of 3,179 inventories at 0.25 seconds per API call, with multiple pages per inventory, took several days. Each machine wrote to its own local `htr_cache/` directory with an independent `_manifest.json` tracking per-inventory completion status.

`merge_caches.py` reconciled the two caches. For the 301 inventories both machines fetched, the script compared non-empty page counts and kept the copy with more pages. In practice all overlapping inventories had identical page counts. Both manifests were preserved as `_manifest_windows.json` and `_manifest_mac.json` for provenance.

`verify_manifest.py` checked each cache against its manifest, checking that every inventory marked "completed" had the expected number of non-empty `.txt` files on disk. Its outputs (`verify_merged_windows.json`, `verify_merged_mac.json`) are consumed by `ner_extract_corpus.py` to skip inventories with known integrity issues.

## §3.3.3 The 98% completeness threshold

Approximately 2% of pages across the corpus contain valid PAGE XML with no extractable text. These are not retrieval failures; the Transkribus HTR model produced XML placeholders but recognized no handwriting (blank pages, heavily damaged pages, or pages the HTR model could not process). The completeness fraction for each inventory was calculated as `non_empty_pages / expected_pages` from the manifest.

The empirical distribution of completeness fractions is bimodal. Most inventories cluster above 98%. A separate group of 346 inventories ranges from 39% to 90%. The 98% threshold sits at the gap between these two populations. Below 98%, it is ambiguous whether the inventory is usable. The missing pages could be concentrated in a section containing most of the names, or scattered harmlessly. The threshold removes this ambiguity.

## §3.4 Train/dev/test split construction

The `ner_split.json` file was curated according to the following procedure. All 339 evaluation inventories (the intersection of Transkribus and VeleHanden) were pooled. Each inventory was assigned to a notary by range-joining against the finding aid. The pool was then split 60/20/20 into train (204), dev (62), and test (73), stratified by notary: within each notary's inventories, approximately 60% went to train, 20% to dev, 20% to test. This ensures every notary's handwriting and scribal conventions appear in the training partition while maintaining strict document-level separation. Of the 26 notaries in the evaluation corpus, 17 appear in the test set. The random seed was 42. The split was generated once and frozen; the same `ner_split.json` was used for all four model evaluations and for all downstream analyses that reference the test set.

## §3.5.2 Training data: why 0.88 (not 0.80)

`create_training_data.py` defaults to `--threshold 0.80` and was run with `--threshold 0.88` for the thesis. The threshold was raised after manual inspection of the 0.80 output revealed systematic false alignments. At 0.80, short VeleHanden names like "Jan" or "Pieter" matched particles and function words in the HTR text that happened to score above threshold. Names of two or three characters matched HTR fragments with no connection to person names. These false alignments injected noise into the BIO tags. Tokens that are not part of any name received B-PER or I-PER labels, teaching the model to extract non-names.

At 0.88, these false alignments dropped sharply. Overall yield (the number of matched spans) decreased by roughly 15%, but manual inspection of 200 random matches at 0.88 showed fewer than 5% were false alignments, compared to an estimated 15-20% at 0.80. The tradeoff favored precision in training signal over volume. 0.90 was also tested but dropped too many genuine matches where HTR corruption or VeleHanden normalization introduced minor spelling differences.

Both the 0.80 and 0.88 CoNLL files (`training_data.conll` and `training_data_088.conll`) are preserved for reproducibility.

## §3.5.2 Training data cleaning (`clean_training_data.py`)

Four post-processing operations target artifacts of the fuzzy-matching alignment. The rates below are from actual script output on the thesis training data.

**A. Trailing punctuation (7.8%).** The fuzzy matcher occasionally extends an entity span to include a trailing period, comma, or colon when the punctuation character does not push the score below threshold. The cleaner strips trailing punctuation from entity-final tokens.

**B. Trailing function words (1.5%).** Spans sometimes capture a trailing function word ("en", "ende", "als", "door", "voor", "met", "die", "dat", "het") when the word falls within the fuzzy match window. The cleaner strips these from entity-final position.

**C. Trailing particles on short entities (1.0%).** Entities of three or fewer tokens sometimes end with a particle that belongs to the surrounding text rather than the name. The cleaner removes these for short entities only, to avoid stripping legitimate prefixes ("van", "de") from longer names where they are part of the family name.

**D. Detectable false positives (0.2%).** Strings beginning with digits, religious formulae ("in den name gods"), and other patterns identifiable as non-names by rule. Removed entirely from the training data.

## §3.5.2 Training hyperparameters

The adapted spaCy model was trained with the following configuration, determined by the constraints of the local machine (Apple M-series, 8GB RAM):

- Base model: `nl_core_news_lg` with tok2vec layer frozen
- Max steps: 20,000
- Patience: 1,600 steps (training stops if dev F1 does not improve for 1,600 consecutive steps)
- Dropout: 0.1
- Batch size: 8 (reduced from spaCy's default of 16-64 due to 8GB RAM constraint)
- Batcher: compounding (start=256, stop=1,500)
- Dev evaluation limit: 5,000 examples subsampled per evaluation for speed
- Best checkpoint: step 7,000 (dev BIO F1 = 74.69%)
- Training stopped: step 8,000 (patience exhausted)
- Random seed: 42
- Token chunking: 250 tokens per training instance (longer sequences were split at sentence boundaries)

The 250-token chunk size balances two considerations. Shorter chunks lose cross-sentence context. Longer chunks exceed spaCy's efficient processing range for transition-based NER and increase memory pressure on the 8GB machine.

## §3.5.3 Evaluation protocol

The headline F1 numbers in the thesis (0.680 adapted, 0.302 off-the-shelf spaCy, etc.) were computed without notary-name or formulaic-term filtering. The evaluation compares all unique lowercased NER predictions against all unique lowercased VeleHanden names per inventory.

`ner_baseline.py` does support two optional filters, which were tested in separate runs:

**Notary-name filtering.** Removes predictions matching the notary's own name (exact match or substring of the notary name from the VeleHanden inventory summary). Without this filter, the notary's name would appear on every page as a correctly extracted person entity, slightly inflating precision.

**Formulaic stoplist.** A set of 28 legal/procedural terms that off-the-shelf spaCy misclassifies as person names: "den comparant", "den requirant", "den testateur", "de testatrice" (legal terms for the appearing party), "den edele hove", "den hove van holland" (court/authority references), "notaris publicq", "juffr", "de heer" (titles), and common HTR noise ("sijn", "sy get", "fo").

In the separate test run, the adapted model's results with and without filtering were:
- RAW: F1 0.6868
- FILTERED: F1 0.6886

The difference is 0.0018. The reason filtering has negligible impact on the adapted model is that the model learned to reject all 28 formulaic terms on its own during training. The filtering summary showed zero formulaic terms removed from the adapted model's output. The only removals were 1,222 notary-name mentions across 73 inventories (0.66% of predictions). For off-the-shelf spaCy, filtering had a larger effect (27,818 formulaic terms and 21,637 notary mentions removed), because the off-the-shelf model had no exposure to notarial language during training and routinely misclassified formulaic terms as person names.

The adapted model's complete rejection of formulaic terms shows how domain adaptation worked. The model encountered these terms thousands of times during training, consistently labeled as non-entities, and learned to stop extracting them.

## §3.5.3 Why `token_sort_ratio`, not `fuzz.ratio`

`fuzz.token_sort_ratio` alphabetically sorts the tokens of both strings before comparing. "Jan Bakker" and "Bakker Jan" score 100 rather than the ~70 that `fuzz.ratio` would assign. This accommodates a common NER behavior where entity boundaries capture name components in a different order than VeleHanden recorded them (e.g., the model extracts "Bakker Jan" from HTR where VeleHanden recorded "Jan Bakker").

This means that coincidentally reordered names ("Jan Pieter" vs "Pieter Jan") also score higher than they should. In practice this is a minor issue for Dutch notarial names because the naming convention follows a rigid structural sequence (given name, patronymic, prefix, family name), so genuine reorderings are rare. The evaluation at 0.85 already filters most coincidental matches.

## §3.6.3 Orthographic variation thresholds

The orthographic variation analysis in `rq2_variation_decomposed_v3.py` uses two thresholds:

- **Surface similarity:** `token_sort_ratio >= 0.85` to identify candidate near-duplicates
- **Family name check:** `fuzz.ratio >= 80` between the family-name components of two candidate pairs

The family name check is applied after decomposition. Two entity strings might score 0.87 at the surface level, but if their family name components score below 80 (e.g., "bakker" vs "boekman"), the pair is rejected as a coincidental match between different people who share a given name. This is the three-way classification producing the 35.6% genuine / 50.5% same-given-different-person / 13.9% coincidental breakdown reported in §4.2.3.

## §3.6.4 External recognition thresholds

`external_recognition_decomp.py` matches extracted entities against SAA Person Reconstructions and ECARTICO.

**SAA decomposed matching.** Extracted names are decomposed into given name and family name via the name parser, then matched against SAA Person Reconstructions at the component level.

Initial thresholds tested were `fuzz.ratio >= 0.80` for given name and `fuzz.ratio >= 0.65` for family name. The low family name threshold was chosen to accommodate orthographic variation. Manual inspection of matches at 0.65 found a high rate of false matches driven by short family names: "Bos" matching "Bok", "Wit" matching "Wil". These are different people.

Final thresholds were `0.85 / 0.85` for both given and family name. At 0.85, the false-match rate in manual inspection dropped below 5%. The higher threshold is conservative (it misses some genuine matches with heavily variant spellings) but the analysis reports match rates as upper bounds, so conservative thresholds strengthen the argument.

The SAA's 1,917 Person Reconstructions produced 58 apparent matches at 0.85/0.85 in a 10,000-entity sample. Manual verification of all 58 classified every one as a short-string collision (three-character given names matching by coincidence). The true decomposed match rate is 0/10,000. The three genuine SAA matches reported in §4.2.5 came from the surface-level matching run (`token_sort_ratio >= 0.85`), not the decomposed matching.

**ECARTICO matching.** ECARTICO's export provides names as single strings without decomposed fields, so decomposed matching was not possible. The ECARTICO results in §4.2.5 use surface-level matching (`token_sort_ratio >= 0.85`), consistent with the SAA surface-level matching run. ECARTICO serves as a corroborating test: a domain-specific authority file whose temporal and geographic scope overlaps with the corpus, testing whether the pattern of match rates across name structure types holds for a different authority database.

## §3.6.5 Stratified NER recall

The recall-by-structure analysis in `recall_by_structure.py` decomposes VeleHanden ground-truth names from the 73 test inventories using the same parser as `decompose_names_v2.py`, then calculates recall per structure type. The VeleHanden names are parsed, not the NER predictions, because this looks at "of the names that exist in the ground truth with structure X, what proportion did the model find?" The NER predictions are matched against VeleHanden at the same 0.85 `token_sort_ratio` threshold used in the main evaluation.

## §3.7.1 Relation indicator dictionary

The dictionary contains 81 indicators across five relation types:

- Spouse: 28 indicators ("sijn huisvrouw", "zijn huijsvrouw", "huisvrouw van", "gehuwd met", "echtgenote van", etc.)
- Child: 19 indicators ("zoon van", "soon van", "dochter van", "sijn soon", "zijn dochter", etc.)
- Sibling: 17 indicators ("sijn broeder", "zijn broer", "sijn suster", "broeder van", etc.)
- Widow: 15 indicators ("weduwe van", "weduwe wijlen", "weduwe van wijlen", "wede van", "de weduwe van wijlen", etc.)
- Widower: 2 indicators ("weduwnaar van", "weduwnaar van wijlen")

The dictionary started from the MiSS indicator set (Ranjbar-Sahraei & Efremova, n.d.) and was expanded by running `check_coverage.py`, which extracts the text between consecutive NER-detected person pairs and counts frequent between-texts that did not match any existing indicator. Recurring relational phrases were reviewed manually, and those encoding genuine familial relationships were added. Archaic and variant spellings of the same phrase (e.g., "sijn" vs "zijn", "huisvrouw" vs "huijsvrouw" vs "huysvrouw") were included as separate entries rather than handled by fuzzy matching, because exact matching against a comprehensive dictionary is faster and more predictable than fuzzy matching against a smaller one.

## §3.7.1 The 150-character between-text cap

The text between two person entities is checked against the indicator dictionary. The cap of 150 characters is applied to the raw between-text before matching. Without a cap, two person names appearing on the same page but in different deeds could be paired, producing false relations between unrelated people. The 150-character limit was determined by measuring the length distribution of between-texts that matched indicators: 95% of genuine matches had between-text shorter than 100 characters. The cap was set at 150 to accommodate multi-line formulae where HTR line breaks inject extra whitespace, while staying well below the typical distance between names in different deeds.

## §3.7.1 Gender plausibility filter

For spouse relations only, the script checks whether both entities carry the same gendered patronymic suffix. If both carry masculine suffixes (-sz, -zoon, -sen) or both carry feminine suffixes (-dochter, -dr), the relation is rejected. Same-gender marriage was not legally recognized in the early modern Netherlands, so a same-gender spouse pair is a false match. Pairs where neither name carries a gendered suffix, or where only one does, pass by default. The filter runs after relation extraction and before output.

## §3.7.1 Skip-one pairing

The relation extraction examines not only consecutive person-entity pairs on a page but also pairs separated by one intervening entity. This captures cases where a witness or co-signer was detected between two related persons. For example, on a page containing entities [A, B, C], the script checks pairs (A,B), (B,C), and (A,C). The between-text for the skip-one pair (A,C) includes entity B's text, which is checked against the same indicator dictionary with the same 150-character cap.

## §3.7.2 Component classification criteria

`rq3_network_stats_v3.py` classifies network components (connected clusters of three or more nodes) by two criteria:

**Given-name ratio ≤ 0.50.** If half or fewer of the entity strings in a cluster have distinct given names (after parsing), the cluster could plausibly represent orthographic variants of a small number of people. For example, a cluster of six nodes with only two distinct given names ("johanna" and "pieter") is more likely to be spelling variants than six unrelated individuals.

**Max degree ≤ 15.** No node in the cluster should be connected to more than 15 other nodes. This threshold was determined empirically using `check_degree_gap.py`, which computes the degree distribution of all clusters meeting the given-name criterion. The distribution showed a clear gap between tight variant clusters (max degree 2-12) and large collision-driven clusters (max degree 30+). The threshold of 15 sits in the gap. Clusters meeting both criteria are classified as "collapsible" (likely orthographic fragmentation). Clusters meeting only the given-name criterion but exceeding the degree threshold are classified as "patronymic artifacts" (collision-driven).

## §3.7 Cross-notary relation counts

Of 41,791 distinct edges: 40,140 connect entities appearing in the same inventory only; 1,651 connect entities appearing in different inventories. Of those 1,651 cross-inventory edges, 242 connect entities appearing under different notaries, and 1,409 connect entities appearing in different inventories of the same notary. The 242 cross-notary relations represent connections that hierarchical description fragments across series with no mechanism to reconstruct.

## §3.6.2 Name parser: patronymic suffix lists

The parser (`decompose_names_v2.py`) distinguishes two categories of patronymic suffixes:

**Reliable suffixes** (unambiguously encode filiation): `-sz`, `-szen`, `-szoon`, `-zoon`, `-szn`, `-sdr`, `-dochter`, `-dgtr`, `-dogter`. A token ending in one of these is always classified as a patronymic.

**Ambiguous suffixes** (could be patronymic or hereditary surname): `-sen`, `-ssen`, `-se`, `-sse`. A token ending in one of these is classified as a patronymic only if the stem (the part before the suffix) matches a known given name. The given-name vocabulary is built from the corpus itself: the first token of every multi-token entity is collected, and tokens appearing more than 50 times are treated as known given names. This means "Jansen" is classified as a patronymic (stem "Jan" is a known given name) but "Mertens" is treated as a family name if "Mert" does not appear as a given name in the corpus.

**Female patronymic markers** (subset of reliable): `-dochter`, `-dgtr`, `-dogter`, `-sdr`, `-dr`. These are used for the gender-marker analysis in §4.2.2.

## §4.2.5 The `ner_extractions_slim.csv` file

Multiple analysis scripts (`rq2_analysis.py`, `decompose_names_v2.py`, `match_ecartico.py`, `rq3_clustering_v2.py`) consume a "slim" version of the NER extractions that drops the character-offset columns (`start_char`, `end_char`, `entity_label`) needed only by the relation extraction script. The slim file was created manually by column-dropping. The repo includes `make_slim_extractions.py` to formalize this step. The slim file retains `inventory_number`, `page_number`, and `entity_text`.

## Computational environment

All analyses except the BERTje and Flair evaluations were conducted on a local macOS machine (Apple M-series, 8GB RAM). BERTje and Flair evaluations were run on Kaggle (GPU environment, `kaggle_ner_eval_v2.py`). Key library versions: Python 3.12.8, spaCy 3.8.11, rapidfuzz 3.14.3, NetworkX 3.x, matplotlib 3.x. The adapted model was trained locally using `spacy train` with the configuration above.
