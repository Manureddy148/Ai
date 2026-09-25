"""Business Entity Resolution — end-to-end baseline pipeline.

data -> normalisation -> blocking (TF-IDF char n-gram top-k) -> pair features
     -> LightGBM matcher (GroupKFold OOF) -> F0.5-tuned threshold -> output TSVs

Run (Kaggle cell or shell):
    python er_pipeline.py --data-dir <dir containing train/ and test/> --out-dir output

The data dir is auto-detected under /kaggle/input and the working dir if omitted.
No external lookups are used; only the provided train/test files.
"""
import argparse
import os
import re
import sys
import time
import unicodedata
from glob import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError:  # Kaggle usually has it; install on the fly otherwise
    os.system(f"{sys.executable} -m pip install -q rapidfuzz")
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

import lightgbm as lgb

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_data_dir(explicit=None):
    if explicit:
        return explicit
    for root in ["/kaggle/input", ".", "..", "dataset"]:
        hits = glob(os.path.join(root, "**", "train_source1.tsv"), recursive=True)
        if hits:
            # .../dataset/train/train_source1.tsv -> .../dataset
            return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError("train_source1.tsv not found; pass --data-dir")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    gt = None
    gt_path = os.path.join(d, f"{split}_ground_truth.tsv")
    if os.path.exists(gt_path):
        g = read_tsv(gt_path)
        gt = {
            r.source1_entity_id: set(x for x in r.matched_entity_ids.split(",") if x)
            for r in g.itertuples(index=False)
        }
    return s1, pd.concat([s2, s3], ignore_index=True), gt


def profile(name, s1, other, gt):
    log(f"== {name}: S1={len(s1):,}  S2+S3={len(other):,}")
    log(f"   S1 countries: {s1.country.value_counts().to_dict()}")
    log(f"   S2/S3 countries: {other.country.value_counts().to_dict()}")
    if gt is not None:
        sizes = np.array([len(v) for v in gt.values()])
        log(f"   singletons: {np.mean(sizes == 0):.3f}  mean matches: {sizes.mean():.2f}"
            f"  max: {sizes.max()}")
    print(s1.head(3).to_string(), "\n", other.head(3).to_string(), flush=True)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
NAME_ABBR = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "pvtltd": "private limited", "intl": "international",
    "mfg": "manufacturing", "svc": "services", "svcs": "services", "bros": "brothers",
    "assoc": "associates", "natl": "national", "tech": "technologies", "grp": "group",
    "ent": "enterprises", "cie": "compagnie", "ste": "societe", "sté": "societe",
}
LEGAL = {
    "corporation", "company", "incorporated", "limited", "private", "llc", "llp", "lp",
    "plc", "pllc", "the", "and", "of", "dba", "pvt", "opc",
    # French legal forms (test-only country, keep generic)
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "societe", "compagnie", "et",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite",
    "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building", "n": "north",
    "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "nr": "near", "opp": "opposite",
    "mg": "mahatma gandhi", "ngr": "nagar", "clny": "colony", "sec": "sector",
    "mkt": "market", "stn": "station", "chowk": "chowk", "marg": "road",
    "salai": "road", "fbg": "faubourg", "chem": "chemin", "imp": "impasse",
}
ADDR_STOP = {"near", "opposite", "behind", "beside", "next", "to", "the", "of", "and",
             "no", "number", "de", "la", "le", "du", "des"}

_punct = re.compile(r"[^a-z0-9 ]+")
_space = re.compile(r"\s+")
_digits = re.compile(r"\d+")
_postal = re.compile(r"\b(\d{5,6})(?:-\d{4})?\b")


def basic(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = re.sub(r"(?<=[a-z])\.(?=[a-z]\.)", "", s)  # p.v.t. -> pvt
    s = _punct.sub(" ", s)
    return _space.sub(" ", s).strip()


def expand(s, table):
    return " ".join(table.get(t, t) for t in s.split())


def norm_name(s):
    return expand(basic(s), NAME_ABBR)


def core_name(n):
    toks = [t for t in n.split() if t not in LEGAL]
    return " ".join(toks) if toks else n


def norm_addr(s):
    return expand(basic(s), ADDR_ABBR)


def prep(df):
    df = df.copy()
    df["name_n"] = df.business_name.map(norm_name)
    df["name_c"] = df.name_n.map(core_name)
    df["addr_n"] = df.business_address.map(norm_addr)
    df["addr_c"] = df.addr_n.map(lambda a: " ".join(t for t in a.split() if t not in ADDR_STOP))
    df["postal"] = df.business_address.map(
        lambda a: (_postal.findall(str(a)) or [""])[-1])
    df["nums"] = df.addr_n.map(lambda a: frozenset(_digits.findall(a)))
    df["country_n"] = df.country.map(basic)
    df["src"] = df.entity_id.str[:2]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Blocking: per-country TF-IDF char n-gram top-k (name) ∪ top-k (name+addr)
# --------------------------------------------------------------------------- #
def topk_sparse(A, B, k, chunk=2000):
    """For each row of A, indices and scores of the top-k rows of B (cosine; rows L2-normed)."""
    idx_out, sc_out = [], []
    BT = B.T.tocsr()
    k = min(k, B.shape[0])
    for i in range(0, A.shape[0], chunk):
        S = (A[i:i + chunk] @ BT).toarray()
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        idx_out.append(part)
        sc_out.append(np.take_along_axis(S, part, axis=1))
    return np.vstack(idx_out), np.vstack(sc_out)


def block(s1, other, k_name=25, k_full=25, k_addr=10):
    fields = {
        "name": (s1.name_c, other.name_c, k_name),
        "full": (s1.name_c + " | " + s1.addr_c, other.name_c + " | " + other.addr_c, k_full),
        "addr": (s1.addr_c, other.addr_c, k_addr),
    }
    vecs = {}
    for key, (a, b, _) in fields.items():
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                            sublinear_tf=True, dtype=np.float32)
        v.fit(pd.concat([a, b]))
        vecs[key] = (v.transform(a), v.transform(b))

    chunks = []
    other_countries = set(other.country_n)
    for c, g1 in s1.groupby("country_n"):
        i1 = g1.index.values
        # open set of countries: fall back to all S2/S3 records for unseen/blank labels
        i2 = other.index.values[other.country_n.values == c] if c in other_countries \
            else other.index.values
        if len(i2) == 0:
            continue
        for key, (_, _, k) in fields.items():
            A, B = vecs[key]
            idx, sc = topk_sparse(A[i1], B[i2], k)
            keep = sc > 0.05
            rows = np.repeat(i1, idx.shape[1]).reshape(idx.shape)
            chunks.append(np.column_stack([rows[keep], i2[idx[keep]]]))
    cand = pd.DataFrame(np.vstack(chunks) if chunks else np.empty((0, 2), int),
                        columns=["i1", "i2"]).drop_duplicates().reset_index(drop=True)
    # cosine similarities for every surviving pair (all three views)
    for key in fields:
        A, B = vecs[key]
        cand[f"cos_{key}"] = np.asarray(
            A[cand.i1.values].multiply(B[cand.i2.values]).sum(axis=1)).ravel()
    log(f"blocking: {len(cand):,} pairs ({len(cand) / max(len(s1), 1):.1f} per S1)")
    return cand


# --------------------------------------------------------------------------- #
# Pair features
# --------------------------------------------------------------------------- #
def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def features(cand, s1, other):
    L, R = s1.loc[cand.i1.values].reset_index(drop=True), other.loc[cand.i2.values].reset_index(drop=True)
    f = cand.copy()
    pairs_name = list(zip(L.name_c, R.name_c))
    pairs_full = list(zip(L.name_n, R.name_n))
    pairs_addr = list(zip(L.addr_c, R.addr_c))
    f["n_ratio"] = [fuzz.ratio(a, b) for a, b in pairs_name]
    f["n_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in pairs_name]
    f["n_tset"] = [fuzz.token_set_ratio(a, b) for a, b in pairs_name]
    f["n_partial"] = [fuzz.partial_ratio(a, b) for a, b in pairs_name]
    f["n_jw"] = [JaroWinkler.similarity(a, b) for a, b in pairs_name]
    f["n_full_ratio"] = [fuzz.ratio(a, b) for a, b in pairs_full]
    f["n_jacc"] = [jacc(a, b) for a, b in pairs_name]
    f["n_first_eq"] = [float(a.split()[:1] == b.split()[:1]) for a, b in pairs_name]
    f["n_init_eq"] = [float("".join(t[0] for t in a.split()) == "".join(t[0] for t in b.split()))
                      for a, b in pairs_name]
    f["a_ratio"] = [fuzz.ratio(a, b) for a, b in pairs_addr]
    f["a_tset"] = [fuzz.token_set_ratio(a, b) for a, b in pairs_addr]
    f["a_partial"] = [fuzz.partial_ratio(a, b) for a, b in pairs_addr]
    f["a_jacc"] = [jacc(a, b) for a, b in pairs_addr]
    lp, rp = L.postal.values, R.postal.values
    f["postal_state"] = np.where((lp == "") | (rp == ""), 0, np.where(lp == rp, 1, -1))
    num_l, num_r = L.nums.values, R.nums.values
    f["num_overlap"] = [len(a & b) / len(a | b) if a and b else -1 for a, b in zip(num_l, num_r)]
    f["num_conflict"] = [float(bool(a) and bool(b) and not (a & b)) for a, b in zip(num_l, num_r)]
    f["len_l"] = L.name_c.str.len().values
    f["len_r"] = R.name_c.str.len().values
    f["alen_l"] = L.addr_c.str.len().values
    f["alen_r"] = R.addr_c.str.len().values
    f["is_s3"] = (R.src.values == "S3").astype(np.int8)
    # context: how this pair compares with the other candidates of the same S1 / same S2-S3 record
    for col in ["cos_full", "n_tset", "a_tset"]:
        f[f"{col}_gap1"] = f[col] - f.groupby("i1")[col].transform("max")
        f[f"{col}_gap2"] = f[col] - f.groupby("i2")[col].transform("max")
    f["rank1"] = f.groupby("i1")["cos_full"].rank(ascending=False, method="first")
    f["rank2"] = f.groupby("i2")["cos_full"].rank(ascending=False, method="first")
    f["n_cand1"] = f.groupby("i1")["i2"].transform("size")
    f["n_cand2"] = f.groupby("i2")["i1"].transform("size")
    return f


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def f05(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def decide(f, prob, thr):
    """Threshold + each S2/S3 record goes to at most one S1 (S1 is deduplicated)."""
    d = f[["i1", "i2"]].copy()
    d["p"] = prob
    d = d[d.p >= thr].sort_values("p", ascending=False).drop_duplicates("i2")
    return d


def to_lists(d, s1, other, col="i2"):
    out = {e: [] for e in s1.entity_id}
    ids1, ids2 = s1.entity_id.values, other.entity_id.values
    for a, b in zip(d.i1.values, d[col].values):
        out[ids1[a]].append(ids2[b])
    return out


def macro_f05(lists, gt, keys):
    return float(np.mean([f05(set(lists.get(k, [])), gt.get(k, set())) for k in keys]))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--folds", type=int, default=5)
    args, _ = ap.parse_known_args()  # tolerate Jupyter's -f argument

    data_dir = find_data_dir(args.data_dir)
    log(f"data dir: {data_dir}")
    tr1, tr2, gt = load_split(data_dir, "train")
    te1, te2, _ = load_split(data_dir, "test")
    profile("train", tr1, tr2, gt)
    profile("test", te1, te2, None)

    tr1, tr2, te1, te2 = prep(tr1), prep(tr2), prep(te1), prep(te2)

    # ---- train ----
    ctr = block(tr1, tr2)
    ftr = features(ctr, tr1, tr2)
    id1, id2 = tr1.entity_id.values, tr2.entity_id.values
    ftr["y"] = [int(id2[b] in gt.get(id1[a], ())) for a, b in zip(ftr.i1.values, ftr.i2.values)]
    total_true = sum(len(v) for v in gt.values())
    log(f"blocking recall ceiling: {ftr.y.sum() / max(total_true, 1):.4f}"
        f"  (pos={ftr.y.sum():,} / true={total_true:,})")

    feats = [c for c in ftr.columns if c not in ("i1", "i2", "y")]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=-1)
    oof = np.zeros(len(ftr))
    iters = []
    for k, (a, b) in enumerate(GroupKFold(args.folds).split(ftr, groups=ftr.i1)):
        m = lgb.train(params, lgb.Dataset(ftr.iloc[a][feats], ftr.y.iloc[a]), 2000,
                      valid_sets=[lgb.Dataset(ftr.iloc[b][feats], ftr.y.iloc[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(ftr.iloc[b][feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {k}: best_iter={m.best_iteration}")

    best_thr, best = 0.5, -1
    for thr in np.arange(0.20, 0.96, 0.025):
        s = macro_f05(to_lists(decide(ftr, oof, thr), tr1, tr2), gt, tr1.entity_id.values)
        if s > best:
            best_thr, best = thr, s
    log(f"OOF macro F0.5 = {best:.4f} at threshold {best_thr:.3f}")
    cand_lists = to_lists(ftr[["i1", "i2"]], tr1, tr2)
    log(f"OOF F0.5 if every candidate were predicted: "
        f"{macro_f05(cand_lists, gt, tr1.entity_id.values):.4f} (sanity)")

    model = lgb.train(params, lgb.Dataset(ftr[feats], ftr.y), int(np.mean(iters) * 1.1))
    imp = pd.Series(model.feature_importance("gain"), feats).sort_values(ascending=False)
    print(imp.head(15).to_string(), flush=True)

    # ---- test ----
    cte = block(te1, te2)
    fte = features(cte, te1, te2)
    pte = model.predict(fte[feats])
    matches = to_lists(decide(fte, pte, best_thr), te1, te2)
    cands = to_lists(fte[["i1", "i2"]], te1, te2)

    os.makedirs(args.out_dir, exist_ok=True)
    for fname, col, lists in [("matching_results.tsv", "matched_entity_ids", matches),
                              ("candidate_pairs.tsv", "candidate_entity_ids", cands)]:
        path = os.path.join(args.out_dir, fname)
        with open(path, "w", newline="") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for e in te1.entity_id.values:
                fh.write(f"{e}\t{','.join(dict.fromkeys(lists[e]))}\n")
        log(f"wrote {path}")
    n_match = sum(bool(v) for v in matches.values())
    log(f"test: {n_match:,}/{len(te1):,} S1 entities matched; "
        f"by country: {te1.assign(m=[bool(matches[e]) for e in te1.entity_id]).groupby('country').m.mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
