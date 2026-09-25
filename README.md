# Amazon ML Challenge: Business Entity Resolution

Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`) must be linked: for every Source 1 entity, predict the Source 2 / Source 3 ids that describe the same real-world business, or leave it empty (singleton).
Training data cover the US and India; the test set additionally contains France, an unseen country, and every test S1 entity must appear in the output.
The metric is macro F0.5 per S1 entity (singletons score 1.0 only when predicted empty); no external data lookup, models must be MIT/Apache-2.0 licensed and at most 8B parameters.

The solution is a single script: text normalisation, per-country TF-IDF char n-gram blocking, hand-crafted pair features, a LightGBM matcher validated with GroupKFold, and a `(threshold, relative-ratio)` decision rule tuned on out-of-fold macro F0.5. Details are in `Documentation_template.md` and `code/business_entity_resolution/README.md`.

## Folder structure

```
.
├── README.md                          this file
├── Documentation_template.md          methodology write-up (submission document)
├── code/
│   └── business_entity_resolution/
│       ├── README.md                  pipeline description and run instructions
│       ├── requirements.txt           numpy, pandas, scipy, scikit-learn, lightgbm, rapidfuzz
│       └── src/
│           ├── run_pipeline.py        entry point (train -> tune -> test -> TSVs)
│           ├── er/                    package: normalize, blocking, features, model
│           └── legacy/er_pipeline_v1.py  first single-file version
├── dataset/                           not committed (see Dataset below)
│   ├── train/                         train_source{1,2,3}.tsv, train_ground_truth.tsv
│   └── test/                          test_source{1,2,3}.tsv
├── output/
│   ├── README.md
│   ├── matching_results.tsv           generated, not committed (source1_entity_id, matched_entity_ids)
│   └── candidate_pairs.tsv            generated, not committed (source1_entity_id, candidate_entity_ids)
├── notebooks/
│   └── kaggle_run.ipynb               end-to-end Kaggle runner: data download, pipeline, validation, submission zip
├── utils/
│   ├── validate_submission.py         stdlib-only checker for the output format rules
│   ├── score.py                       stdlib-only macro F0.5 scorer (per-country / per-size breakdowns)
│   └── make_synthetic_dataset.py      synthetic train/test set in the challenge format (smoke test)
├── docs/
│   ├── Architecture.pdf               architecture and design document (rendered)
│   └── architecture/                  its HTML sources: sections/*.html, style.css, build_doc.py
└── 6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf
```

The final submission zip contains `output/{matching_results.tsv,candidate_pairs.tsv}`, `code/business_entity_resolution/{src/,README.md,requirements.txt}` and `Documentation_template.md`; the Kaggle notebook builds it.

## How to run

### Kaggle notebook (recommended)

Upload `notebooks/kaggle_run.ipynb` to Kaggle, enable internet, and run all cells. The notebook installs the dependencies, downloads and unzips the dataset from Google Drive (or uses an attached Kaggle input dataset), fetches the `er` package, `run_pipeline.py` and the utilities from this repository (branch `claude/wizardly-cori-snj0nx`), runs the pipeline, validates the two output files and zips the submission package under `/kaggle/working`.

### Locally

```bash
git clone https://github.com/Manureddy148/Ai && cd Ai
# unzip the dataset so that dataset/train/ and dataset/test/ exist
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --out-dir output --work-dir work
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### Smoke test without the real data

```bash
python3 utils/make_synthetic_dataset.py /tmp/synth
python code/business_entity_resolution/src/run_pipeline.py --data-dir /tmp/synth --out-dir /tmp/synth_out --work-dir /tmp/synth_work --train-s1 2000 --val-s1 800 --distractor-frac 0.5
python3 utils/validate_submission.py --matching /tmp/synth_out/matching_results.tsv \
    --candidate /tmp/synth_out/candidate_pairs.tsv --test-dir /tmp/synth/test
python3 utils/score.py --pred /tmp/synth_out/matching_results.tsv \
    --truth /tmp/synth/test_ground_truth_hidden.tsv --source1 /tmp/synth/test/test_source1.tsv --by-size
```

The synthetic set (3,000 train / 1,500 test S1 entities, US + India + France in test) runs end-to-end in about 30 s and exercises every stage, the validator and the scorer.

### Rebuilding the architecture document

`python3 docs/architecture/build_doc.py` assembles `docs/architecture/sections/*.html` with `style.css` and renders `docs/Architecture.pdf` with headless Chromium (two passes, so the table of contents carries page numbers). Requires `pip install pymupdf`.

The run log prints the blocking recall ceiling and the out-of-fold macro F0.5 with the selected decision parameters. `utils/score.py --pred <predictions.tsv> --truth dataset/train/train_ground_truth.tsv --source1 dataset/train/train_source1.tsv --by-size` scores any prediction file in ground-truth format.

## Dataset

The dataset is not committed (see `.gitignore`); download it from Google Drive and unzip it into `dataset/`:

- https://drive.google.com/file/d/162gyZK2G7xnie6N3whTVFlwAP4frHFy9/view?usp=sharing
- https://drive.google.com/file/d/1bukugde70Drs9bHw8nr5oSr72ZPHDHzc/view?usp=sharing
