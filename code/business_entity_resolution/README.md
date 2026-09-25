# Business Entity Resolution: baseline pipeline

Single script: `src/er_pipeline.py`. Only the provided train/test files are used (no external lookups).

## Pipeline
1. **Normalise**: ASCII-fold, lowercase, `&`→`and`, collapse dotted acronyms (`p.v.t.`→`pvt`),
   split explicit alias markers (`dba`, `d/b/a`, `t/a`, `a.k.a.`, `formerly known as`) into name
   variants, expand name/address abbreviations (Pvt→private, Rd→road, …), strip legal suffixes
   (Ltd, LLC, SARL, …) to get a core name, extract the postal code, the digit runs and the leading
   (house) number of the address.
2. **Blocking**: for each country (an open set; unseen labels fall back to all records),
   TF-IDF char 2–4-grams, top-k by core name (k=25), name+address (k=25) and address (k=10), plus
   a fourth view: same postal code, top-10 by name cosine. Pairs with cosine ≤ 0.05 are dropped.
   Union → `candidate_pairs.tsv`.
3. **Features**: TF-IDF cosines of the three views, rapidfuzz ratio / token-sort / token-set /
   partial / Jaro-Winkler, token Jaccard, first-token and initials agreement, best alias-pair score
   (`n_alias_best`, `has_alias`), acronym-vs-expansion flag, postal-code and house-number agreement
   (ternary), street-number overlap/conflict, string lengths, source flag, and contextual features:
   each pair's gap from the best `cos_full` / `n_tset` / `a_tset` among the same S1's and the same
   S2/S3 record's candidates, its rank by `cos_full` on both sides, and the candidate counts.
4. **Matcher**: LightGBM with 5-fold GroupKFold grouped by S1 entity. Decoding: each S2/S3 record is
   assigned to the S1 that scores it highest (Source 1 is deduplicated); a pair is kept if
   `p >= thr` **and** `p >= ratio * best_p` within its (S1, source) group. The pair
   `(thr, ratio)` is swept on out-of-fold predictions (`thr` in 0.20…0.95 step 0.025,
   `ratio` in {0, 0.6, 0.7, 0.8, 0.9, 1.0}) to maximise **macro F0.5 including singletons**.

## Run
From the repository root:
```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/er_pipeline.py --data-dir dataset --out-dir output
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
`--data-dir` is the folder containing `train/` and `test/`. In a Kaggle notebook:
`!python er_pipeline.py --out-dir /kaggle/working/output` (the data dir is auto-detected under
`/kaggle/input`), or use `notebooks/kaggle_run.ipynb`, which also packages the submission zip.

The log prints the blocking recall ceiling (overall, per country, per source), the blocking oracle
and all-candidates floor, and the out-of-fold macro F0.5 with the selected `(thr, ratio)`, which is
the validation score.
