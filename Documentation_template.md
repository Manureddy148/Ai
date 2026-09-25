# Business Entity Resolution — Methodology Write-up

Amazon ML Challenge: Business Entity Resolution

## Team / Overview

| Field | Value |
|---|---|
| Team | TBD: fill team name |
| Members | TBD: fill member names |
| Submission | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| Code | `code/business_entity_resolution/src/er_pipeline.py` (single script) |
| Model | LightGBM binary classifier on hand-crafted pair features |
| External data | None (only the provided train/test TSVs are read) |

The task is to link every Source 1 (S1) business record to the Source 2 / Source 3 (S2/S3) records that describe the same real-world business, or to declare it a singleton. Records carry a free-text name, a free-text address and a country label. The training data cover the United States and India; the test data additionally contain France, which is never seen during training. The metric is macro-averaged F0.5 over S1 entities, where singletons score 1.0 only when the prediction is empty. Our solution is a classical record-linkage pipeline: aggressive text normalisation, per-country TF-IDF character-n-gram blocking, a supervised pairwise matcher (LightGBM) trained on ground-truth pairs with GroupKFold out-of-fold (OOF) validation, and a decision rule tuned directly on the OOF macro-F0.5 including singletons.

## 1. Methodology

End-to-end flow, as implemented in `er_pipeline.py`:

```
 train/{source1,source2,source3,ground_truth}.tsv      test/{source1,source2,source3}.tsv
                 |                                                   |
                 v                                                   v
 +-------------------------------------------------------------------------------+
 | 1. LOAD      read_tsv (quoting=3, dtype=str, no NA parsing); S2+S3 concatenated|
 +-------------------------------------------------------------------------------+
                 |
                 v
 +-------------------------------------------------------------------------------+
 | 2. NORMALISE prep(): NFKD ascii-fold, lowercase, & -> and, dotted-acronym fix, |
 |    alias-marker split (dba / d/b/a / t/a / a.k.a. / formerly), abbreviation    |
 |    expansion, legal-suffix stripping (core name), address stop-word removal,   |
 |    postal-code, street-number and house-number extraction                     |
 +-------------------------------------------------------------------------------+
                 |
                 v
 +-------------------------------------------------------------------------------+
 | 3. BLOCK     block(): per-country TF-IDF char_wb 2-4gram, three fuzzy views    |
 |    (core name k=25, name|addr k=25, addr k=10) + same-postal-code view         |
 |    (name cosine, k=10), union, cosine > 0.05,                                  |
 |    unseen country -> compare against all S2/S3 records                        |
 |    => candidate pairs (i1, i2) + cos_name / cos_full / cos_addr               |
 +-------------------------------------------------------------------------------+
                 |
                 v
 +-------------------------------------------------------------------------------+
 | 4. FEATURES  features(): string similarities, token overlaps, alias/acronym   |
 |    matches, postal/house/street-number agreement, lengths, source flag, and   |
 |    contextual gap/rank/crowding features relative to the competing candidates |
 |    of the same S1 and same S2/S3 record                                       |
 +-------------------------------------------------------------------------------+
                 |
        +--------+---------+
        | train            | test
        v                  v
 +-----------------+   +----------------------------------------------------------+
 | 5. MATCHER      |   | 6. DECISION  decode_table() + decide(): each S2/S3 record |
 | LightGBM,       |-->|    goes to its highest-probability S1; keep the pair if  |
 | GroupKFold(5)   |   |    p >= thr AND p >= ratio * best p in its (S1, source)  |
 | by S1, early    |   +----------------------------------------------------------+
 | stopping, OOF   |                        |
 | (thr, ratio)    |                        v
 | sweep on        |   +----------------------------------------------------------+
 | macro-F0.5,     |   | 7. OUTPUT  matching_results.tsv (matches),               |
 | refit on all    |   |    candidate_pairs.tsv (all blocked pairs); one row per  |
 +-----------------+   |    test S1 entity, ids de-duplicated, tab-separated      |
                       +----------------------------------------------------------+
```

The same `prep -> block -> features` path is applied to train and test, so the feature distribution the model sees at inference time is the one it was trained on. The matcher is trained only on pairs that survive blocking (the ground-truth pairs that blocking misses are neither positives nor negatives for the model; they are counted in the recall ceiling reported in the log).

## 2. Data understanding & preprocessing

### 2.1 Data profile

All files are tab-separated with columns `entity_id`, `business_name`, `business_address`, `country`. Entity ids are prefixed `S1-`, `S2-`, `S3-`. Ground truth maps each S1 id to a comma-separated list of S2/S3 ids (empty for singletons). The `profile()` step logs, per split, the record counts, country distribution per source, singleton fraction and the mean/max number of matches per S1 entity; these numbers drive the choice of top-k in blocking and are recorded in Section 8. Source 1 is treated as de-duplicated (each S2/S3 record belongs to at most one S1 entity), which the decision rule in Section 6 enforces.

Noise patterns observed and addressed:

- Case, diacritics and Unicode variants (`Société`, `SOCIETE`, full-width characters).
- Ampersands and `@` (`A&B Traders` vs `A and B Traders`).
- Dotted acronyms (`P.V.T. Ltd`, `L.L.C.`).
- Explicit aliases inside one field (`Acme Holdings LLC dba Acme Pizza`, `X t/a Y`, `formerly known as`), where one source carries the legal name and another the trade name.
- Legal-form variation and omission (`Acme Inc`, `Acme, Incorporated`, `Acme`).
- Address abbreviations (`Rd`/`Road`, `St`/`Street`, `Blvd`/`Bd`, `Ste`/`Suite`), Indian address idioms (`Nr`, `Opp`, `MG Rd`, `Ngr`, `Clny`, `Sec`, `Marg`, `Salai`), French address idioms (`Fbg`, `Chem`, `Imp`, articles `de/la/le/du/des`).
- Missing or partial fields (empty addresses, missing postal codes), handled with explicit "unknown" states rather than imputed values.
- Blank or unseen country labels, handled in blocking (Section 3.4).

### 2.2 Normalisation rules (as implemented)

`basic(s)`:

1. `unicodedata.normalize("NFKD")`, drop non-ASCII code points, lowercase.
2. `&` -> ` and `, `@` -> ` at `.
3. Collapse dotted acronyms: a dot between letters followed by another letter-dot is removed (`p.v.t.` -> `pvt`).
4. Replace every character outside `[a-z0-9 ]` with a space; collapse whitespace.

Alias markers are handled on the raw string before `basic()` destroys `/` and `.`: the regular expression `_ALIAS` matches `dba`, `d/b/a`, `d.b.a.`, `doing business as`, `trading as`, `t/a`, `a/k/a`, `a.k.a.`, `also known as` and `formerly (known as)` (a bare `aka` is deliberately not matched, as it can be a legitimate name token). `name_n` is the normalised name with the markers removed (legal and trade names concatenated, so blocking keeps its recall), and `name_alts` is the list of the individual core-normalised variants used by the alias features.

`norm_name = expand(basic(name), NAME_ABBR)` with the abbreviation table

| Token | Expansion | Token | Expansion |
|---|---|---|---|
| corp | corporation | co | company |
| inc | incorporated | ltd | limited |
| pvt | private | pvtltd | private limited |
| intl | international | mfg | manufacturing |
| svc / svcs | services | bros | brothers |
| assoc | associates | natl | national |
| tech | technologies | grp | group |
| ent | enterprises | cie | compagnie |
| ste / sté | societe | | |

`core_name` removes every token in `LEGAL` and falls back to the full normalised name when nothing remains:

```
corporation company incorporated limited private llc llp lp plc pllc the and of dba pvt opc
sarl sas sasu sa eurl sci snc societe compagnie et
```

The second line covers French legal forms (SARL, SAS, SASU, SA, EURL, SCI, SNC, Société, Compagnie, "et") so that the core name is comparable across the three countries even though France is absent from training.

`norm_addr = expand(basic(address), ADDR_ABBR)` with the table

| Token | Expansion | Token | Expansion |
|---|---|---|---|
| rd | road | st / str | street |
| ave / av | avenue | blvd / bd | boulevard |
| dr | drive | ln | lane |
| hwy | highway | pkwy | parkway |
| ct | court | pl | place |
| sq | square | ste | suite |
| apt | apartment | fl / flr | floor |
| bldg | building | n / s / e / w | north / south / east / west |
| ne / nw / se / sw | northeast / … | nr | near |
| opp | opposite | mg | mahatma gandhi |
| ngr | nagar | clny | colony |
| sec | sector | mkt | market |
| stn | station | marg / salai | road |
| fbg | faubourg | chem | chemin |
| imp | impasse | | |

`addr_c` additionally drops the stop words `near opposite behind beside next to the of and no number de la le du des`.

Derived fields:

- `postal`: last match of `\b(\d{5,6})(?:-\d{4})?\b` on the raw address (5-digit US ZIP with optional +4, 6-digit Indian PIN, 5-digit French code); empty string if none.
- `nums`: frozenset of the digit runs in the normalised address (street numbers, block numbers, floor numbers) with the postal code removed, except when the postal-code match is the leading number of the address and does not end it (a 5-digit US house number with the ZIP missing is kept as the house number).
- `house`: the leading digit run of the normalised address (house / plot number), empty string if none.
- `country_n`: `basic(country)`.
- `src`: first two characters of `entity_id` (`S2` or `S3`).

## 3. Candidate generation / blocking

### 3.1 Representation

Three TF-IDF vectorisers (`analyzer="char_wb"`, `ngram_range=(2,4)`, `min_df=2`, `sublinear_tf=True`, float32) are fitted on the concatenation of the S1 and S2/S3 texts for the split being processed, one per view:

| View | S1 text | S2/S3 text | top-k |
|---|---|---|---|
| `name` | `name_c` | `name_c` | 25 |
| `full` | `name_c + " | " + addr_c` | `name_c + " | " + addr_c` | 25 |
| `addr` | `addr_c` | `addr_c` | 10 |
| `postal` (exact-key view) | `name_c`, restricted to records sharing the S1 postal code | `name_c` | 10 |

Character n-grams within word boundaries are robust to typos, token reordering and residual abbreviation differences that the tables do not cover. TF-IDF rows are L2-normalised, so a sparse dot product is a cosine similarity.

### 3.2 Per-country top-k retrieval

For each country value in S1, the S2/S3 index is restricted to records with the same normalised country. `topk_sparse` multiplies chunks of S1 rows against the transposed S2/S3 matrix, densifies the chunk, and takes the top-k columns per row with `np.argpartition`. The chunk size is derived from the S2/S3 block size so that the dense chunk stays bounded (about 20M cells, a few hundred MB) regardless of how large a country block or the whole-corpus unseen-country fallback is. Pairs whose cosine is at or below 0.05 are discarded. A fourth, orthogonal view is added inside the same per-country loop: S1 and S2/S3 records sharing the same postal code (ZIP / PIN) are compared by name cosine and the top-10 per S1 are kept, which recovers true matches with heavily noisy names inside one postal block. The views are unioned and de-duplicated into a `(i1, i2)` candidate table, and the cosine of every surviving pair is then recomputed for the three fuzzy views (`cos_name`, `cos_full`, `cos_addr`), in slices of 200k pairs to bound memory, so that a pair retrieved only by the address view still carries its name cosine as a feature.

The union of a name-only view, a name-plus-address view and an address-only view is what gives recall: the name view finds records whose address was entered differently or is missing, the address view finds records whose name is a trading name or a heavily abbreviated variant, and the combined view ranks the pairs where both agree.

### 3.3 Measuring recall ceiling and reduction ratio

On the training split each candidate pair is labelled `y = 1` if the S2/S3 id is in the S1 entity's ground-truth list. The pipeline logs

- blocking recall ceiling = `sum(y) / total ground-truth pairs`, i.e. the fraction of true links that any downstream matcher can still recover;
- the number of S1 entities with at least one true match among their candidates (the metric is macro over S1 entities, so an S1 whose true match was never blocked scores 0.0), and the blocking recall broken down by country and by source (S2 / S3);
- pairs per S1 = `len(cand) / len(S1)`, which together with `len(S1) * len(S2+S3)` gives the reduction ratio `1 - pairs / (|S1| * |S2+S3|)`;
- the macro-F0.5 obtained if every candidate were predicted as a match ("floor" line), which bounds how much the matcher must add over blocking;
- the blocking-oracle macro-F0.5 obtained by predicting exactly the true candidate pairs ("ceiling" line), i.e. the score of a perfect matcher on the blocked pairs.

The same candidate table, converted to id lists, is written verbatim as `candidate_pairs.tsv`, so the file the organisers score for candidate quality is exactly the set the matcher saw.

### 3.4 Open-set countries

`block()` treats country as an open set. If an S1 country value does not occur among the S2/S3 records (an unseen label such as France in a hypothetical split where only S1 carries it, a blank, or a spelling variant), the S1 group is compared against all S2/S3 records instead of an empty block, so no S1 entity is silently dropped. In the actual test set France appears in all three sources, so French S1 records are blocked against French S2/S3 records exactly as US and Indian records are; the vectorisers are refitted on the test texts, so French n-gram vocabulary is learned from the test data itself without any external resource.

## 4. Model architecture & feature engineering

### 4.1 Model

A single LightGBM gradient-boosted tree ensemble with a binary log-loss objective scores each candidate pair:

| Parameter | Value |
|---|---|
| objective | binary |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_child_samples | 40 |
| feature_fraction | 0.8 |
| bagging_fraction / bagging_freq | 0.8 / 1 |
| lambda_l2 | 1.0 |
| max rounds | 2000 with early stopping (patience 100) |

Tree ensembles are a good fit here because the features are heterogeneous (bounded similarity scores, ternary agreement states, counts, ranks) and interact non-linearly (for example, a moderate name score is decisive when the postal code agrees and the street number does not conflict, but not otherwise). The parameter count is in the order of 10^5 split thresholds, far below the 8B limit.

### 4.2 Pair features

All features are computed by `features()` on the normalised fields. `L` and `R` denote the S1 and S2/S3 records of the pair.

| Family | Feature | Definition |
|---|---|---|
| TF-IDF cosine | `cos_name`, `cos_full`, `cos_addr` | Cosine of the L2-normalised char 2-4gram TF-IDF vectors for the three blocking views |
| rapidfuzz, name | `n_ratio` | `fuzz.ratio(L.name_c, R.name_c)` (Indel-based similarity, 0-100) |
| | `n_tsort` | `fuzz.token_sort_ratio` on core names (order-insensitive) |
| | `n_tset` | `fuzz.token_set_ratio` on core names (robust to extra tokens on one side) |
| | `n_partial` | `fuzz.partial_ratio` on core names (substring match) |
| | `n_jw` | Jaro-Winkler similarity on core names (prefix-weighted) |
| | `n_full_ratio` | `fuzz.ratio` on the full normalised names (legal suffixes retained) |
| Token overlap, name | `n_jacc` | Jaccard index of the core-name token sets |
| | `n_first_eq` | 1 if the first tokens of both core names are equal |
| | `n_init_eq` | 1 if the strings of token initials are equal (`ibm` vs `international business machines`) |
| Aliases | `n_alias_best` | Best `fuzz.token_sort_ratio` over every (variant of L) x (variant of R) pair from `name_alts`, so a legal name on one side matches the trade name on the other |
| | `has_alias` | 1 if either record carries an explicit alias marker (more than one name variant) |
| Acronym | `n_acronym` | 1 if one core name is a single token equal to the initials of the other, multi-token core name (`sbi` vs `state bank india`) |
| rapidfuzz, address | `a_ratio`, `a_tset`, `a_partial` | Same as above on `addr_c` |
| Token overlap, address | `a_jacc` | Jaccard index of address token sets |
| Postal agreement | `postal_state` | 0 if either postal code is missing, 1 if equal, -1 if both present and different |
| House number | `house_state` | 0 if either leading address number is missing, 1 if equal, -1 if both present and different |
| Street numbers | `num_overlap` | Jaccard of the digit-run sets (postal code excluded); -1 when either side has no numbers |
| | `num_conflict` | 1 when both sides have numbers and share none |
| Lengths | `len_l`, `len_r`, `alen_l`, `alen_r` | Character lengths of core name and cleaned address on each side (lets the model calibrate similarity scores by string length) |
| Source flag | `is_s3` | 1 if the candidate is an S3 record (sources differ in field quality) |
| Context: gap to best | `cos_full_gap1`, `n_tset_gap1`, `a_tset_gap1` | Feature value minus the maximum of that feature over all candidates of the same S1 entity (0 for the best candidate, negative otherwise) |
| | `cos_full_gap2`, `n_tset_gap2`, `a_tset_gap2` | Same, relative to all candidates of the same S2/S3 record |
| Context: rank | `rank1` | Rank of the pair by `cos_full` among the S1 entity's candidates (1 = best) |
| | `rank2` | Rank of the pair by `cos_full` among the S2/S3 record's candidates |
| Context: crowding | `n_cand1`, `n_cand2` | Number of candidates of the S1 entity / of the S2/S3 record |

The context features are the main departure from a purely pairwise scorer. A pair that is the best candidate for both its S1 entity and its S2/S3 record (`gap1 = gap2 = 0`, `rank1 = rank2 = 1`) is a mutual nearest neighbour; a pair that looks similar in absolute terms but is dominated on either side is much more likely to be a near-duplicate of a different business (chain branches, franchises, common names). Encoding this lets a pairwise model approximate the collective decision that a full assignment would make, while remaining trivially parallel. Feature importances (gain) for the top 15 features are printed after the final fit and are recorded in Section 8.

## 5. Training & validation

1. Candidate pairs are generated on the training split, and labels are read from `train_ground_truth.tsv`.
2. `GroupKFold(n_splits=5)` with `groups = i1` (the S1 row) ensures that all candidates of an S1 entity fall in the same fold, so the contextual features and the threshold are validated on entities the model has not seen, mirroring the test situation.
3. In each fold, LightGBM is trained for up to 2,000 rounds with early stopping (patience 100) on the held-out fold's log-loss, and the held-out predictions are stored as OOF probabilities. The best iteration of each fold is recorded.
4. The decision rule of Section 6 has two parameters, an absolute threshold `thr` and a relative ratio `ratio`. They are swept jointly: `ratio` over `{0.0, 0.6, 0.7, 0.8, 0.9, 1.0}` and `thr` over `0.20, 0.225, …, 0.95` (step 0.025). For each pair the full decision rule is applied to the OOF probabilities, the predicted lists are compared to the ground truth with the official per-entity F0.5 (singletons included: 1.0 if predicted empty, else 0.0), and the macro average over all training S1 entities is computed. The `(thr, ratio)` pair with the highest OOF macro-F0.5 is kept (`ratio = 0` reproduces a flat per-pair threshold, so the sweep can never do worse than threshold-only tuning).
5. The final model is refit on all training pairs with `round(1.1 * mean(best_iteration over folds))` boosting rounds, the usual correction for the larger training set when early stopping is no longer available.

Because `(thr, ratio)` is tuned on OOF predictions of the same model family and feature set, the OOF macro-F0.5 is an honest estimate of the leaderboard score for the seen countries; the unseen-country gap is discussed in Section 7.

## 6. Decision rule

`decode_table(f, prob)` (threshold-independent part, computed once):

1. Sort the scored pairs by probability descending and `drop_duplicates("i2")`: each S2/S3 record is assigned to at most one S1 entity, the one with the highest probability.
2. For every surviving pair record `best`, the maximum probability within its `(S1, source)` group, where source is S2 or S3 (`is_s3`).

`decide(d, thr, ratio)`:

3. Keep pairs with `p >= thr` **and** `p >= ratio * best`, i.e. a pair must be confident in absolute terms and not be dominated by a much stronger candidate of the same S1 from the same source.
4. An S1 entity may keep several S2/S3 records (the ground truth contains multi-record matches), and an S1 entity with no surviving pair is output as a singleton (empty `matched_entity_ids`).

Rationale:

- F0.5 weights precision twice as heavily as recall, so a false positive costs more than a missed link. The `(thr, ratio)` sweep, rather than a fixed 0.5, finds the operating point where this trade-off is optimal on the actual score.
- Singletons contribute 1.0 or 0.0 with nothing in between. A single spurious link on a singleton costs a full point, whereas a missed link on a true match costs only part of a point. This pushes the optimal threshold upward and is why the sweep must include singletons; tuning on matched entities only would over-predict.
- The relative ratio addresses the multi-match case: when an S1 entity has one very confident S2 candidate and a second S2 candidate that is only moderately confident, the second one is far more often a near-duplicate of a different business (a chain branch, a common name) than a second true record. Dropping it when `p < ratio * best` trades a little recall on genuine multi-record entities for precision, which F0.5 rewards. Grouping by source keeps an S3 candidate from being suppressed by a strong S2 candidate, since each source can legitimately contribute its own record. Since the sweep includes `ratio = 0`, the rule is only used when it improves the OOF score.
- The one-S1-per-S2/S3 constraint follows from Source 1 being deduplicated. Removing an S2/S3 record from every S1 except its best one removes the most likely false positives at essentially no recall cost, and the gap/rank features already push the model towards the same choice, so the constraint mostly acts as a safety net on near-ties.
- Matches are a subset of candidates by construction, since only blocked pairs are scored.

## 7. Handling the unseen country (France)

Nothing in the pipeline is country-specific except the blocking partition. The measures that make the model transfer are:

- Language-agnostic normalisation: NFKD folding removes accents, so `Société Générale` and `Societe Generale` become identical; the legal-suffix list contains the common French forms so that the core name is comparable, and the address table covers `bd`, `fbg`, `chem`, `imp` and French articles.
- Character-n-gram blocking with vectorisers refitted on the test data, so French vocabulary is learned at inference time from the provided records only.
- Features that are relative rather than absolute: the gap/rank/crowding features describe how a pair compares with its competitors, which is stable across languages even if the absolute similarity distribution shifts.
- Postal-code extraction that accepts 5- or 6-digit codes and therefore covers French codes without a dedicated rule.
- The open-set fallback in blocking guarantees that every test S1 entity gets candidates even if its country label had no S2/S3 counterpart.

The decision parameters `(thr, ratio)` are tuned on US and Indian entities only; the run log reports the per-country fraction of test S1 entities that received at least one match, which is used as a sanity check that France is neither starved nor over-matched relative to the seen countries.

## 8. Results

| Quantity | Value |
|---|---|
| Train S1 / S2+S3 records | TBD: fill from run log |
| Train singleton fraction | TBD: fill from run log |
| Blocking recall ceiling (train), overall / by country / by source | TBD: fill from run log |
| S1 entities with >= 1 true match blocked | TBD: fill from run log |
| Candidate pairs per S1 (train / test) | TBD: fill from run log |
| Reduction ratio | TBD: fill from run log |
| Macro-F0.5 if all candidates were predicted (floor) | TBD: fill from run log |
| Blocking-oracle macro-F0.5 (ceiling) | TBD: fill from run log |
| Fold best iterations | TBD: fill from run log |
| OOF macro-F0.5 (incl. singletons) | TBD: fill from run log |
| Selected threshold `thr` / relative ratio `ratio` | TBD: fill from run log |
| Test S1 entities matched, by country | TBD: fill from run log |
| Top gain features | TBD: fill from run log |
| Public leaderboard score | TBD: fill from leaderboard |

## 9. Fair play & licensing

- No external data lookup of any kind: the script reads only the provided train and test TSVs; there is no web access, no gazetteer, no company registry and no pretrained language model. (The only network action is an optional `pip install rapidfuzz` fallback in case the package is missing, which installs a library, not data.)
- Dependencies and licences: LightGBM (MIT), scikit-learn (BSD-3-Clause), rapidfuzz (MIT), NumPy (BSD), pandas (BSD), SciPy (BSD). All permit use in the challenge.
- Model size: a LightGBM ensemble of at most 2,000 trees with 63 leaves, well under the 8B-parameter limit; no pretrained weights are loaded.
- Test labels are never used; the threshold and the number of boosting rounds come from training-set cross-validation only.

## 10. Reproduction

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python src/er_pipeline.py --data-dir ../../dataset --out-dir ../../output   # --folds 5 by default
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

`--data-dir` must contain `train/` and `test/`; if omitted, the script searches `/kaggle/input`, the working directory, its parent and `dataset/` for `train_source1.tsv`. In a Kaggle notebook: `!python er_pipeline.py --out-dir /kaggle/working/output`. The log prints every quantity listed in Section 8. Runtime is dominated by the rapidfuzz features over the candidate table; LightGBM uses all cores (`n_jobs=-1`). Bagging and feature subsampling make the model stochastic at the level of the last decimals; set `seed` in `params` for bit-exact reproduction.

## 11. Limitations & future work

- Pairwise scoring only: the gap/rank features approximate but do not enforce a globally consistent assignment. A transitive-consistency step (e.g. resolving S2-S3 pairs that both link to the same S1, or a small bipartite assignment per connected component) could remove residual conflicts.
- A single global `(thr, ratio)`: the optimal operating point may differ per country or per source; per-country parameters (tuned only on seen countries and extrapolated to France) or per-source parameters are a low-risk extension.
- Hand-crafted normalisation: the abbreviation and legal-form tables are finite. A learned pair scorer, for instance a cross-encoder fine-tuned from an Apache-2.0 model well under 8B parameters (e.g. a small multilingual encoder) on the blocked training pairs, used either as an additional LightGBM feature or as a second-stage re-ranker of near-threshold pairs, would generalise to unseen abbreviations and languages.
- Blocking recall: top-k is fixed per view; an adaptive k or a dense-embedding view alongside TF-IDF would raise the ceiling for records whose name and address are both heavily rewritten.
- Absence of a France validation set: the transfer to French records is verified only through match-rate sanity checks. Held-out validation on a pseudo-unseen country (train on US, validate on India, and vice versa) would quantify the expected drop and guide the threshold margin.
