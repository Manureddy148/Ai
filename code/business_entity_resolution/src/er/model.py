"""LightGBM matcher, per-S1 decoding and vectorised macro-F0.5 scoring."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .log import log

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, n_jobs=4, seed=42)


def train(X, y, groups, rounds=3000, patience=100):
    """Train with an internal group-held-out early-stopping split (20% of S1 groups)."""
    rng = np.random.default_rng(0)
    g = np.unique(groups)
    hold = np.isin(groups, rng.choice(g, size=max(1, len(g) // 5), replace=False))
    dtr = lgb.Dataset(X[~hold], y[~hold])
    dva = lgb.Dataset(X[hold], y[hold])
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)])
    log(f"model: best_iter={m.best_iteration}, train pairs={(~hold).sum():,}, holdout pairs={hold.sum():,}, "
        f"pos rate={y.mean():.4f}")
    imp = pd.Series(m.feature_importance("gain"), X.columns).sort_values(ascending=False)
    log("feature importance (gain, top 12): " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(12).items()))
    return m


def decode_table(i1, i2, p):
    """Assign each S2/S3 record to its best-scoring S1; record the best p per S1."""
    d = pd.DataFrame({"i1": i1, "i2": i2, "p": p.astype(np.float32)})
    d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("i2")
    d["best"] = d.groupby("i1").p.transform("max")
    return d.reset_index(drop=True)


def decide(d, thr, ratio):
    return d[(d.p >= thr) & (d.p >= ratio * d.best)]


def macro_f05(pred_i1, pred_y, n_true, n_s1):
    """pred_i1: S1 index per predicted pair; pred_y: 1 if the pair is true; n_true: true
    matches per S1 (array over all n_s1 entities). Singletons score 1 iff nothing predicted."""
    n_pred = np.bincount(pred_i1, minlength=n_s1).astype(np.float64)
    tp = np.bincount(pred_i1, weights=pred_y, minlength=n_s1)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(n_pred > 0, tp / n_pred, 0.0)
        R = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
    return float(f.mean())


def tune(d, y_pair, n_true, n_s1):
    """Grid-search (thr, ratio) on a decode table with truth labels; returns best triple."""
    best = (-1.0, 0.5, 0.0)
    for ratio in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        for thr in np.arange(0.10, 0.96, 0.025):
            m = (d.p.to_numpy() >= thr) & (d.p.to_numpy() >= ratio * d.best.to_numpy())
            s = macro_f05(d.i1.to_numpy()[m], y_pair[m], n_true, n_s1)
            if s > best[0]:
                best = (s, float(thr), ratio)
    log(f"tune: macro F0.5={best[0]:.4f} at thr={best[1]:.3f} ratio={best[2]}")
    return best
