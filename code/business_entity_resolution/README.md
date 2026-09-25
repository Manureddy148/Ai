# Business Entity Resolution: baseline pipeline

Single script: `src/er_pipeline.py`. Only the provided train/test files are used (no external lookups).

## Pipeline
1. **Normalise**: ASCII-fold, lowercase, `&`→`and`, expand name/address abbreviations
   (Pvt→private, Rd→road, …), strip legal suffixes (Ltd, LLC, SARL, …) to get a core name,
   extract postal code and street numbers.
2. **Blocking**: for each country (an open set; unseen labels fall back to all records),
   TF-IDF char 2–4-grams, top-k by core name (25), name+address (25) and address (10). Union → `candidate_pairs.tsv`.
3. **Features**: TF-IDF cosines, rapidfuzz ratio/token-set/partial/Jaro-Winkler, token Jaccard,
   postal code agreement, street-number overlap/conflict, and each pair's gap from the best score/rank among the same S1's and the same S2/S3 record's candidates.
4. **Matcher**: LightGBM with 5-fold GroupKFold grouped by S1 entity; the decision threshold is tuned on
   out-of-fold predictions for **macro F0.5 including singletons**. Each S2/S3 record is
   assigned to at most one S1 entity (Source 1 is deduplicated).

## Run
```bash
pip install -r requirements.txt
python src/er_pipeline.py --data-dir <folder containing train/ and test/> --out-dir output
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
In a Kaggle notebook: `!python er_pipeline.py --out-dir /kaggle/working/output`
(the data dir is auto-detected under `/kaggle/input`).

The log prints the blocking recall ceiling and the out-of-fold F0.5, which is the validation score.
