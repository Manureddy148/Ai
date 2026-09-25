# Business Entity Resolution: pipeline

Only the provided train/test files are used (no external lookups, no network access at run time).

## Layout

```
src/
├── run_pipeline.py        entry point: train -> tune -> test -> output TSVs
├── er/
│   ├── normalize.py       name/address normalisation (unidecode ASCII folding, abbreviation
│   │                      tables, legal-suffix and TLD stripping, postal / house / digit extraction)
│   ├── blocking.py        hash-key blocking (name, address and house-number keys, per-key cap)
│   ├── features.py        pairwise features (rapidfuzz cpdist scorers, address agreement, context)
│   ├── model.py           LightGBM matcher, per-S1 decoding, vectorised macro-F0.5, grid tuning
│   └── log.py
└── legacy/er_pipeline_v1.py   first single-file version (TF-IDF top-k blocking); kept for reference
```

## Pipeline

1. **Normalise** (`er.normalize.prep`, process pool): `unidecode` folds Devanagari / Tamil /
   Kannada transliterations and accents to ASCII; lowercase; `&`->`and`; dotted acronyms
   collapsed; abbreviations expanded; legal suffixes (Pvt, Ltd, LLC, SARL, ...) and web TLDs
   (`.com`) removed to obtain the core name; addresses lose filler tokens (`near`, `null`, ...);
   postal code (5-6 digits), house number (first token containing a digit) and the digit runs
   are extracted.
2. **Block** (`er.blocking.block`): every record emits keys, all prefixed with the country:
   first two name tokens, 8-char no-space name prefix, sorted 4-char token prefixes (word
   reordering), house number + street token, house number + each of the last two address
   tokens, first name token + house number, or name token + first address token when there is
   no house number. Keys are hashed to int64; keys shared by more than `--cap` (150) S2/S3
   records are dropped; pairs are formed by a sorted-array join. The number of shared keys is a
   feature.
3. **Features** (`er.features.features`): rapidfuzz `cpdist` (multithreaded) ratio /
   token-sort / token-set / partial / Jaro-Winkler on core names, token-set / ratio / partial on
   addresses; postal and house-number agreement (ternary), digit-run overlap / conflict, lengths,
   token counts, 6-char prefix equality, source flag, non-ASCII flag; context within the S1
   group: gaps to the best candidate (token-set, address token-set, ratio), rank and candidate
   count.
4. **Train** (`er.model.train`): LightGBM binary classifier on a sample of `--train-s1`
   (150k) S1 entities blocked against their true matches plus `--distractor-frac` (10%) of the
   S2/S3 pool; early stopping on a held-out 20% of S1 groups.
5. **Tune**: `--val-s1` (40k) disjoint S1 entities blocked against the **full** training pool
   (realistic candidate density). Each S2/S3 record is assigned to its best-scoring S1; a pair is
   kept if `p >= thr` and `p >= ratio * best_p(S1)`. `(thr, ratio)` is grid-searched on the
   vectorised macro F0.5 (singletons included).
6. **Test**: all test S1 entities blocked against the full test pool; features and predictions
   in chunks of `--chunk-pairs` complete S1 groups; decode; write `matching_results.tsv` and
   `candidate_pairs.tsv` (every pair the model scored).

## Run

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data-dir <dir with train/ and test/> --out-dir output --work-dir work
python3 ../../utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir <dir>/test
```

`--work-dir` caches the normalised frames (parquet), the model and the tuned thresholds;
`--skip-train` reuses them. Resources for the full challenge data (2.2M / 10.3M train,
1.7M / 10M test records): 4 CPUs, ~12 GB RAM, ~2 hours.

The log prints: ground-truth statistics, blocking recall (train sample and validation), the
validation oracle F0.5 (blocking ceiling), the tuned `(thr, ratio)` with its macro F0.5 and
per-country scores, feature importances, and test match rates per country.
