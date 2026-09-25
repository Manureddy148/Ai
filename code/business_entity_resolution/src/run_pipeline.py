#!/usr/bin/env python3
"""End-to-end entity resolution: train on a sample of the training split, tune the
decision rule on a held-out sample scored against the FULL training pool, then run
the test split in chunks and write output/matching_results.tsv + candidate_pairs.tsv.

    python src/run_pipeline.py --data-dir dataset --out-dir output --work-dir work

Intermediate artefacts (prepped frames, model, tuned thresholds) are cached in --work-dir
so a crashed run can resume with --skip-train.
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from er.blocking import block          # noqa: E402
from er.features import FEATURES, features  # noqa: E402
from er.log import log                 # noqa: E402
from er.model import decide, decode_table, macro_f05, train, tune  # noqa: E402
from er.normalize import prep          # noqa: E402
import lightgbm as lgb                 # noqa: E402


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_prepped(path, work, tag, workers):
    cache = os.path.join(work, f"{tag}.parquet")
    if os.path.exists(cache):
        log(f"{tag}: loading cache {cache}")
        return pd.read_parquet(cache)
    df = read_tsv(path) if isinstance(path, str) else pd.concat([read_tsv(p) for p in path], ignore_index=True)
    log(f"{tag}: {len(df):,} rows read; normalising")
    out = prep(df, workers=workers)
    del df
    out.to_parquet(cache, index=False)
    log(f"{tag}: normalised and cached")
    return out


def pair_codes(i1, i2):
    return i1.astype(np.int64) * (1 << 32) + i2.astype(np.int64)


def feats_chunked(pairs, s1, other, workers, chunk_pairs):
    """Features for all pairs, computed per block of complete i1 groups; returns DataFrame."""
    pairs = pairs.sort_values(["i1", "i2"], kind="stable").reset_index(drop=True)
    i1 = pairs.i1.to_numpy()
    bounds = np.flatnonzero(np.diff(i1)) + 1          # group starts
    outs, start = [], 0
    while start < len(pairs):
        target = start + chunk_pairs
        if target >= len(pairs):
            end = len(pairs)
        else:
            end = bounds[np.searchsorted(bounds, target)] if np.searchsorted(bounds, target) < len(bounds) else len(pairs)
        outs.append(features(pairs.iloc[start:end], s1, other, workers))
        log(f"  features: {end:,}/{len(pairs):,} pairs")
        start = end
    return pairs, pd.concat(outs, ignore_index=True)


def predict_chunked(model, pairs, s1, other, workers, chunk_pairs):
    pairs = pairs.sort_values(["i1", "i2"], kind="stable").reset_index(drop=True)
    i1 = pairs.i1.to_numpy()
    bounds = np.flatnonzero(np.diff(i1)) + 1
    p = np.empty(len(pairs), dtype=np.float32)
    start = 0
    while start < len(pairs):
        target = start + chunk_pairs
        k = np.searchsorted(bounds, target)
        end = len(pairs) if target >= len(pairs) or k >= len(bounds) else bounds[k]
        X = features(pairs.iloc[start:end], s1, other, workers)
        p[start:end] = model.predict(X[FEATURES], num_threads=workers)
        log(f"  predict: {end:,}/{len(pairs):,} pairs")
        start = end
    return pairs, p


def write_lists(path, col, s1_ids, i1_sorted, id_strings):
    """One row per S1 entity (file order); id_strings aligned with i1_sorted (sorted by i1)."""
    bounds = np.searchsorted(i1_sorted, np.arange(len(s1_ids) + 1))
    with open(path, "w", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for k, e in enumerate(s1_ids):
            fh.write(e + "\t" + ",".join(id_strings[bounds[k]:bounds[k + 1]]) + "\n")
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-s1", type=int, default=150_000, help="S1 entities sampled for training")
    ap.add_argument("--val-s1", type=int, default=40_000, help="S1 entities held out for threshold tuning")
    ap.add_argument("--distractor-frac", type=float, default=0.10)
    ap.add_argument("--cap", type=int, default=150, help="max S2/S3 records per blocking key")
    ap.add_argument("--chunk-pairs", type=int, default=8_000_000)
    ap.add_argument("--skip-train", action="store_true", help="reuse work/model.txt and work/tuned.json")
    ap.add_argument("--max-test-s1", type=int, default=0, help="debug: limit test S1 rows")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    D = args.data_dir
    model_path, tuned_path = os.path.join(args.work_dir, "model.txt"), os.path.join(args.work_dir, "tuned.json")

    if not (args.skip_train and os.path.exists(model_path) and os.path.exists(tuned_path)):
        # ------------------------------------------------------------------ train
        s1 = load_prepped(f"{D}/train/train_source1.tsv", args.work_dir, "train_s1", args.workers)
        pool = load_prepped([f"{D}/train/train_source2.tsv", f"{D}/train/train_source3.tsv"], args.work_dir, "train_pool", args.workers)
        gt = read_tsv(f"{D}/train/train_ground_truth.tsv")
        s1_pos = pd.Index(s1.entity_id).get_indexer(gt.source1_entity_id)
        assert (s1_pos >= 0).all()
        lists = gt.matched_entity_ids.str.split(",")
        n_true = np.zeros(len(s1), dtype=np.int32)
        sizes = lists.map(lambda x: 0 if x == [""] else len(x)).to_numpy()
        n_true[s1_pos] = sizes
        t_i1 = np.repeat(s1_pos, sizes)
        t_ids = [x for l in lists for x in l if x]
        t_i2 = pd.Index(pool.entity_id).get_indexer(t_ids)
        ok = t_i2 >= 0
        log(f"ground truth: {len(gt):,} S1, {len(t_ids):,} matches ({(~ok).sum():,} ids not in pool), "
            f"singletons={np.mean(sizes == 0):.3f}, mean matches={sizes.mean():.2f}")
        t_i1, t_i2 = t_i1[ok], t_i2[ok]
        true_codes = np.sort(pair_codes(t_i1, t_i2))
        del gt, lists

        rng = np.random.default_rng(42)
        perm = rng.permutation(len(s1))
        A, B = np.sort(perm[:args.train_s1]), np.sort(perm[args.train_s1:args.train_s1 + args.val_s1])

        # training pool: true matches of A + a random fraction of everything else
        in_A = np.zeros(len(s1), bool); in_A[A] = True
        match_rows = np.unique(t_i2[in_A[t_i1]])
        rand = rng.random(len(pool)) < args.distractor_frac
        sub = np.flatnonzero(rand); sub = np.union1d(sub, match_rows)
        log(f"train sample: {len(A):,} S1, pool subset {len(sub):,} rows ({len(match_rows):,} true matches)")
        s1A, poolS = s1.iloc[A].reset_index(drop=True), pool.iloc[sub].reset_index(drop=True)
        pairs = block(s1A, poolS, cap=args.cap)
        codes = pair_codes(A[pairs.i1.to_numpy()], sub[pairs.i2.to_numpy()])
        y = np.isin(codes, true_codes).astype(np.int8)
        n_true_A = n_true[A]
        log(f"train blocking recall: {y.sum():,}/{n_true_A.sum():,} = {y.sum() / max(n_true_A.sum(), 1):.4f}; "
            f"pairs/S1={len(pairs) / len(A):.1f}")
        pairs, X = feats_chunked(pairs, s1A, poolS, args.workers, args.chunk_pairs)
        y = np.isin(pair_codes(A[pairs.i1.to_numpy()], sub[pairs.i2.to_numpy()]), true_codes).astype(np.int8)
        model = train(X[FEATURES], y, pairs.i1.to_numpy())
        model.save_model(model_path)
        del X, pairs, y, s1A, poolS; gc.collect()

        # ------------------------------------------------------------------ validation vs FULL pool
        s1B = s1.iloc[B].reset_index(drop=True)
        pairs = block(s1B, pool, cap=args.cap)
        codesB = pair_codes(B[pairs.i1.to_numpy()], pairs.i2.to_numpy())
        yB = np.isin(codesB, true_codes)
        n_true_B = n_true[B]
        log(f"val blocking recall: {yB.sum():,}/{n_true_B.sum():,} = {yB.sum() / max(n_true_B.sum(), 1):.4f}; "
            f"pairs/S1={len(pairs) / len(B):.1f}; S1 with all matches blocked: "
            f"{np.mean(np.bincount(pairs.i1.to_numpy()[yB], minlength=len(B)) == n_true_B):.4f}")
        pairs, pB = predict_chunked(model, pairs, s1B, pool, args.workers, args.chunk_pairs)
        yB = np.isin(pair_codes(B[pairs.i1.to_numpy()], pairs.i2.to_numpy()), true_codes).astype(np.float64)
        d = decode_table(pairs.i1.to_numpy(), pairs.i2.to_numpy(), pB)
        y_d = np.isin(pair_codes(B[d.i1.to_numpy()], d.i2.to_numpy()), true_codes).astype(np.float64)
        oracle = macro_f05(pairs.i1.to_numpy()[yB > 0], yB[yB > 0], n_true_B, len(B))
        log(f"val oracle F0.5 (blocking ceiling) = {oracle:.4f}")
        score, thr, ratio = tune(d, y_d, n_true_B, len(B))
        kept = decide(d, thr, ratio)
        yk = np.isin(pair_codes(B[kept.i1.to_numpy()], kept.i2.to_numpy()), true_codes)
        for c in np.unique(s1B.country):
            m = (s1B.country.to_numpy() == c)
            idx = np.flatnonzero(m)
            sel = np.isin(kept.i1.to_numpy(), idx)
            log(f"  val {c}: F0.5={macro_f05(kept.i1.to_numpy()[sel], yk[sel].astype(float), np.where(m, n_true_B, 0), len(B)) * len(B) / m.sum() - (~m).sum() / m.sum():.4f}")
        json.dump({"thr": thr, "ratio": ratio, "val_f05": score, "oracle": oracle}, open(tuned_path, "w"))
        del s1, pool, pairs, d, kept; gc.collect()

    model = lgb.Booster(model_file=model_path)
    tuned = json.load(open(tuned_path))
    thr, ratio = tuned["thr"], tuned["ratio"]
    log(f"using thr={thr} ratio={ratio} (val F0.5={tuned['val_f05']:.4f})")

    # ---------------------------------------------------------------------- test
    te1 = load_prepped(f"{D}/test/test_source1.tsv", args.work_dir, "test_s1", args.workers)
    if args.max_test_s1:
        te1 = te1.iloc[:args.max_test_s1].reset_index(drop=True)
    pool = load_prepped([f"{D}/test/test_source2.tsv", f"{D}/test/test_source3.tsv"], args.work_dir, "test_pool", args.workers)
    log(f"test: S1={len(te1):,} countries={te1.country.value_counts().to_dict()}; pool={len(pool):,}")
    pairs = block(te1, pool, cap=args.cap)
    pairs, p = predict_chunked(model, pairs, te1, pool, args.workers, args.chunk_pairs)
    np.save(os.path.join(args.work_dir, "test_pairs_i1.npy"), pairs.i1.to_numpy())
    np.save(os.path.join(args.work_dir, "test_pairs_i2.npy"), pairs.i2.to_numpy())
    np.save(os.path.join(args.work_dir, "test_pairs_p.npy"), p)
    d = decode_table(pairs.i1.to_numpy(), pairs.i2.to_numpy(), p)
    kept = decide(d, thr, ratio).sort_values(["i1", "p"], ascending=[True, False], kind="stable")
    pool_ids = pool.entity_id.to_numpy(dtype=object)
    s1_ids = te1.entity_id.tolist()
    write_lists(os.path.join(args.out_dir, "matching_results.tsv"), "matched_entity_ids", s1_ids,
                kept.i1.to_numpy(), pool_ids[kept.i2.to_numpy()].tolist())
    write_lists(os.path.join(args.out_dir, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids,
                pairs.i1.to_numpy(), pool_ids[pairs.i2.to_numpy()].tolist())
    n_match = np.bincount(kept.i1.to_numpy(), minlength=len(te1))
    cn = te1.country.to_numpy()
    log(f"test: {np.mean(n_match > 0):.4f} of S1 matched; mean matches={n_match.mean():.2f}; "
        f"by country: { {c: round(float(np.mean(n_match[cn == c] > 0)), 4) for c in np.unique(cn)} }")


if __name__ == "__main__":
    main()
