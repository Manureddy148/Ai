#!/usr/bin/env python3
"""
Standard-library-only validator for Amazon ML Challenge (Business Entity
Resolution) submission files.

Mirrors the checks of the official validator:
  * exact headers, exactly two tab-separated columns per row
  * every test source-1 entity present exactly once, no unknown / duplicate rows
  * every listed id has an S2-/S3- prefix, exists in the test set, and is not
    repeated within a list; no empty tokens, no quoting characters
  * cross-check: matched ids must be a subset of candidate ids (an error when
    --candidate is given, since the rules require it)

Usage:
  python3 utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
import argparse
import csv
import os
import sys

MAX_SHOWN = 50           # cap on issues printed
QUOTE_CHARS = ('"', "'")


def load_test_ids(test_dir):
    """Return (set of S1 ids, set of S2/S3 ids) read from the test TSVs."""
    s1, others = set(), set()
    for fname, target in (("test_source1.tsv", s1),
                          ("test_source2.tsv", others),
                          ("test_source3.tsv", others)):
        path = os.path.join(test_dir, fname)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader, None)
            if not header or header[0] != "entity_id":
                sys.exit("ERROR: %s must start with an 'entity_id' column" % path)
            for row in reader:
                if row and row[0]:
                    target.add(row[0].strip())
    return s1, others


def validate_file(path, list_col, s1_ids, other_ids):
    """
    Validate one submission file. Returns (issues, mapping, stats) where
    mapping is {source1_entity_id: [ids...]} for the cross-check.
    """
    issues, mapping = [], {}
    seen = set()
    total_ids = 0

    with open(path, newline="", encoding="utf-8") as fh:
        raw_lines = fh.read().splitlines()

    if not raw_lines:
        return ["%s: file is empty" % path], mapping, {}

    # --- header ---------------------------------------------------------
    expected = ["source1_entity_id", list_col]
    header = raw_lines[0].split("\t")
    if header != expected:
        issues.append("%s: header must be exactly %s, got %s"
                      % (path, "\t".join(expected), raw_lines[0]))

    # --- rows -----------------------------------------------------------
    for lineno, line in enumerate(raw_lines[1:], start=2):
        if line == "":                       # tolerate trailing blank line
            continue
        cols = line.split("\t")
        if len(cols) != 2:
            issues.append("%s line %d: expected 2 tab-separated columns, got %d"
                          % (path, lineno, len(cols)))
            continue
        sid, id_list = cols
        if any(q in line for q in QUOTE_CHARS):
            issues.append("%s line %d: quoting characters are not allowed"
                          % (path, lineno))
        if sid in seen:
            issues.append("%s line %d: duplicate source1_entity_id %s"
                          % (path, lineno, sid))
        seen.add(sid)
        if sid not in s1_ids:
            issues.append("%s line %d: unknown source1_entity_id %s"
                          % (path, lineno, sid))

        ids = []
        if id_list != "":
            for tok in id_list.split(","):
                if tok == "" or tok != tok.strip():
                    issues.append("%s line %d: empty or whitespace-padded token in list"
                                  % (path, lineno))
                    continue
                if not (tok.startswith("S2-") or tok.startswith("S3-")):
                    issues.append("%s line %d: id %s must have S2-/S3- prefix"
                                  % (path, lineno, tok))
                elif tok not in other_ids:
                    issues.append("%s line %d: id %s not found in test set"
                                  % (path, lineno, tok))
                if tok in ids:
                    issues.append("%s line %d: duplicate id %s within list"
                                  % (path, lineno, tok))
                ids.append(tok)
        mapping[sid] = ids
        total_ids += len(ids)

    # --- completeness ---------------------------------------------------
    missing = s1_ids - seen
    for sid in sorted(missing):
        issues.append("%s: missing row for test source1_entity_id %s" % (path, sid))

    rows = len(mapping)
    stats = {
        "rows": rows,
        "with_match": sum(1 for v in mapping.values() if v),
        "total_ids": total_ids,
        "avg_len": (total_ids / rows) if rows else 0.0,
    }
    return issues, mapping, stats


def print_stats(label, stats):
    if not stats:
        return
    print("[%s] rows=%d  entities_with_>=1_id=%d  total_ids=%d  avg_list_len=%.3f"
          % (label, stats["rows"], stats["with_match"],
             stats["total_ids"], stats["avg_len"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, help="matching_results.tsv")
    ap.add_argument("--candidate", help="candidate_pairs.tsv (optional)")
    ap.add_argument("--test-dir", required=True, help="dir with test_source{1,2,3}.tsv")
    args = ap.parse_args()

    s1_ids, other_ids = load_test_ids(args.test_dir)
    print("Test set: %d S1 entities, %d S2/S3 entities" % (len(s1_ids), len(other_ids)))

    issues, warnings = [], []
    m_issues, matches, m_stats = validate_file(
        args.matching, "matched_entity_ids", s1_ids, other_ids)
    issues.extend(m_issues)
    print_stats("matching", m_stats)

    if args.candidate:
        c_issues, cands, c_stats = validate_file(
            args.candidate, "candidate_entity_ids", s1_ids, other_ids)
        issues.extend(c_issues)
        print_stats("candidate", c_stats)
        # Cross-check: matches must be a subset of candidates (a rule, so an error).
        for sid, ids in matches.items():
            cset = set(cands.get(sid, []))
            extra = [i for i in ids if i not in cset]
            if extra:
                issues.append("%s: matched ids not in candidate list: %s"
                              % (sid, ",".join(extra)))
    else:
        warnings.append("no --candidate file given; subset-of-candidates rule not checked")

    for w in warnings[:MAX_SHOWN]:
        print("WARNING: " + w)
    if len(warnings) > MAX_SHOWN:
        print("... %d warnings total" % len(warnings))

    if not issues:
        print("PASS")
        return 0
    print("FAILED: %d issue(s) found" % len(issues))
    for n, msg in enumerate(issues[:MAX_SHOWN], start=1):
        print("%d. %s" % (n, msg))
    if len(issues) > MAX_SHOWN:
        print("... showing %d of %d issues" % (MAX_SHOWN, len(issues)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
