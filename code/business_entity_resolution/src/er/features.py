"""Pairwise features for candidate pairs, computed on chunks of pairs.

String similarities use rapidfuzz's multithreaded element-wise `cpdist`; the remaining
features are vectorised with numpy. Context features (gap to the best candidate of the
same S1, rank, candidate count) are computed within the chunk, so chunks must contain
complete S1 groups (the caller splits by i1 ranges).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = ["n_keys", "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "a_tset", "a_ratio", "a_partial",
            "postal_state", "house_state", "num_overlap", "num_conflict", "len_l", "len_r", "alen_l", "alen_r",
            "addr_empty_r", "is_s3", "non_ascii_r", "ntok_l", "ntok_r", "prefix_eq",
            "gap_ntset", "gap_atset", "gap_nratio", "rank1", "n_cand1"]


def _cp(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _tri(l, r):
    l, r = np.asarray(l, dtype=object), np.asarray(r, dtype=object)
    empty = (l == "") | (r == "")
    return np.where(empty, 0, np.where(l == r, 1, -1)).astype(np.int8)


def _num_feats(nl, nr):
    ov = np.empty(len(nl), dtype=np.float32)
    cf = np.zeros(len(nl), dtype=np.int8)
    for k, (a, b) in enumerate(zip(nl, nr)):
        if a and b:
            sa, sb = set(a.split()), set(b.split())
            inter = len(sa & sb)
            ov[k] = inter / len(sa | sb)
            cf[k] = inter == 0
        else:
            ov[k] = -1
    return ov, cf


def features(pairs, s1, other, workers=4):
    """pairs: DataFrame(i1, i2, n_keys) -> feature DataFrame (same row order)."""
    i1, i2 = pairs.i1.to_numpy(), pairs.i2.to_numpy()
    ln = s1.name_c.to_numpy(dtype=object)[i1].tolist()
    rn = other.name_c.to_numpy(dtype=object)[i2].tolist()
    la = s1.addr_c.to_numpy(dtype=object)[i1].tolist()
    ra = other.addr_c.to_numpy(dtype=object)[i2].tolist()
    f = pd.DataFrame({"n_keys": pairs.n_keys.to_numpy()})
    f["n_ratio"] = _cp(ln, rn, fuzz.ratio, workers)
    f["n_tsort"] = _cp(ln, rn, fuzz.token_sort_ratio, workers)
    f["n_tset"] = _cp(ln, rn, fuzz.token_set_ratio, workers)
    f["n_partial"] = _cp(ln, rn, fuzz.partial_ratio, workers)
    f["n_jw"] = _cp(ln, rn, JaroWinkler.normalized_similarity, workers)
    f["a_tset"] = _cp(la, ra, fuzz.token_set_ratio, workers)
    f["a_ratio"] = _cp(la, ra, fuzz.ratio, workers)
    f["a_partial"] = _cp(la, ra, fuzz.partial_ratio, workers)
    f["postal_state"] = _tri(s1.postal.to_numpy(dtype=object)[i1], other.postal.to_numpy(dtype=object)[i2])
    f["house_state"] = _tri(s1.house.to_numpy(dtype=object)[i1], other.house.to_numpy(dtype=object)[i2])
    ov, cf = _num_feats(s1.nums.to_numpy(dtype=object)[i1].tolist(), other.nums.to_numpy(dtype=object)[i2].tolist())
    f["num_overlap"], f["num_conflict"] = ov, cf
    f["len_l"] = np.fromiter((len(x) for x in ln), np.int16, len(ln))
    f["len_r"] = np.fromiter((len(x) for x in rn), np.int16, len(rn))
    f["alen_l"] = np.fromiter((len(x) for x in la), np.int16, len(la))
    f["alen_r"] = np.fromiter((len(x) for x in ra), np.int16, len(ra))
    f["addr_empty_r"] = (f.alen_r == 0).astype(np.int8)
    f["is_s3"] = (other.src.to_numpy()[i2] == "S3").astype(np.int8)
    f["non_ascii_r"] = other.non_ascii.to_numpy()[i2]
    f["ntok_l"] = np.fromiter((x.count(" ") + 1 for x in ln), np.int8, len(ln))
    f["ntok_r"] = np.fromiter((x.count(" ") + 1 for x in rn), np.int8, len(rn))
    f["prefix_eq"] = np.fromiter((a.replace(" ", "")[:6] == b.replace(" ", "")[:6] for a, b in zip(ln, rn)),
                                 np.int8, len(ln))
    # context within the S1 group (chunks hold complete i1 groups)
    g = f.groupby(i1, sort=False)
    f["gap_ntset"] = f.n_tset - g.n_tset.transform("max")
    f["gap_atset"] = f.a_tset - g.a_tset.transform("max")
    f["gap_nratio"] = f.n_ratio - g.n_ratio.transform("max")
    f["rank1"] = g.n_tset.rank(ascending=False, method="first").astype(np.int16)
    f["n_cand1"] = g.n_tset.transform("size").astype(np.int32)
    return f[FEATURES]
