#!/usr/bin/env python3
"""Stdlib-only scorer for the Amazon ML Challenge: Business Entity Resolution.

Usage:
    python3 utils/score.py --pred output/matching_results.tsv --truth dataset/train/train_ground_truth.tsv
                           [--source1 dataset/train/train_source1.tsv] [--by-size]

Both --pred and --truth are TSVs: <source1_entity_id> TAB <comma-separated S2-/S3- ids>.
Metric (as defined by the challenge): macro F0.5 over every S1 entity in the TRUTH file.
  * truth empty (singleton): 1.0 if prediction empty, else 0.0
  * prediction empty, truth non-empty: 0.0
  * tp == 0: 0.0
  * otherwise F0.5 = 1.25 * P * R / (0.25 * P + R)
S1 entities missing from --pred are treated as empty predictions (a warning is printed).
"""
import argparse
import csv
import sys
from collections import defaultdict


def read_lists(path):
    """Read a two-column TSV -> {s1_id: set(ids)}. Header row is skipped; blank second column -> empty set."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            ids = row[1].strip() if len(row) > 1 else ""
            out[row[0].strip()] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def read_country(path):
    """Read a source TSV (entity_id, business_name, business_address, country) -> {entity_id: country}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            out[row["entity_id"].strip()] = (row.get("country") or "").strip() or "UNKNOWN"
    return out


def f05(pred, truth):
    """Per-entity F0.5 exactly as the challenge defines it."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def size_bucket(n):
    return "3+" if n >= 3 else str(n)


def main():
    ap = argparse.ArgumentParser(description="Macro F0.5 scorer for entity-resolution predictions.")
    ap.add_argument("--pred", required=True, help="predictions TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--truth", required=True, help="ground-truth TSV (same format)")
    ap.add_argument("--source1", help="source1 TSV; enables per-country breakdown")
    ap.add_argument("--by-size", action="store_true", help="print F0.5 bucketed by truth list size (0,1,2,3+)")
    args = ap.parse_args()

    pred, truth = read_lists(args.pred), read_lists(args.truth)
    country = read_country(args.source1) if args.source1 else None

    missing = [k for k in truth if k not in pred]
    if missing:
        print(f"WARNING: {len(missing)} truth entities missing from predictions (scored as empty)", file=sys.stderr)
    extra = len(set(pred) - set(truth))
    if extra:
        print(f"WARNING: {extra} predicted entities not in truth (ignored)", file=sys.stderr)

    # Accumulators: overall, per-country and per-size-bucket F0.5 sums / counts, plus micro tp/fp/fn.
    scores, by_country, by_size = [], defaultdict(list), defaultdict(list)
    tp = fp = fn = 0
    singletons = singleton_ok = 0

    for s1, t in truth.items():
        p = pred.get(s1, set())
        s = f05(p, t)
        scores.append(s)
        tp += len(p & t); fp += len(p - t); fn += len(t - p)
        if not t:
            singletons += 1
            singleton_ok += int(not p)
        if country is not None:
            by_country[country.get(s1, "UNKNOWN")].append(s)
        if args.by_size:
            by_size[size_bucket(len(t))].append(s)

    n = len(scores)
    macro = sum(scores) / n if n else 0.0
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0

    print(f"Entities scored      : {n}")
    print(f"Macro F0.5           : {macro:.5f}")
    print(f"Micro precision      : {micro_p:.5f}  (tp={tp}, fp={fp})")
    print(f"Micro recall         : {micro_r:.5f}  (fn={fn})")
    if singletons:
        print(f"Singleton accuracy   : {singleton_ok / singletons:.5f}  ({singleton_ok}/{singletons})")
        print(f"False merges (singl.): {singletons - singleton_ok}")
    else:
        print("Singleton accuracy   : n/a (no singletons in truth)")

    if country is not None:
        print("\nPer-country F0.5:")
        for c in sorted(by_country, key=lambda k: (-len(by_country[k]), k)):
            v = by_country[c]
            print(f"  {c:<12} n={len(v):<7} F0.5={sum(v) / len(v):.5f}")

    if args.by_size:
        print("\nF0.5 by truth list size:")
        for b in ("0", "1", "2", "3+"):
            v = by_size.get(b, [])
            print(f"  size {b:<3} n={len(v):<7} F0.5={(sum(v) / len(v)) if v else float('nan'):.5f}")


if __name__ == "__main__":
    main()
