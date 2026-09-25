"""Hash-key blocking: candidate pairs are S1 x S2/S3 records that share at least one key.

Keys are strings built from the normalised fields (always prefixed with the country so
records never pair across countries), hashed to int64 and joined with numpy. Keys shared
by more than `cap` S2/S3 records are dropped (they carry no discriminative power and
would explode the pair count). The number of distinct keys a pair shares is returned as a
feature (`n_keys`).
"""
import numpy as np
import pandas as pd

from .log import log


def _keys_for(name_c, addr_c, house, country):
    """All blocking keys of one record (list of strings)."""
    c = country
    toks = name_c.split()
    keys = []
    if toks:
        keys.append(f"n2|{c}|{' '.join(toks[:2])}")
        keys.append(f"np|{c}|{name_c.replace(' ', '')[:8]}")
        if len(toks) >= 2:
            keys.append("ns|" + c + "|" + "|".join(sorted(t[:4] for t in toks)[:3]))
    atoks = addr_c.split()
    if house:
        # house number + the first purely alphabetic token after it (street / area)
        after = False
        street = ""
        for t in atoks:
            if after and t.isalpha() and len(t) > 2:
                street = t
                break
            if t == house:
                after = True
        if street:
            keys.append(f"ah|{c}|{house}|{street}")
        # house number + each of the last two alphabetic tokens (city / state)
        alpha = [t for t in atoks if t.isalpha() and len(t) > 2]
        for t in alpha[-2:]:
            keys.append(f"al|{c}|{house}|{t}")
        if toks:
            keys.append(f"nh|{c}|{toks[0]}|{house}")
    elif atoks and toks:
        # no house number: name token + first address token
        keys.append(f"na|{c}|{toks[0]}|{atoks[0]}")
    return keys


def key_table(frame, chunk=500_000):
    """(hash int64, row int32) arrays for every key of every record of `frame`."""
    hs, rows = [], []
    name_c, addr_c, house, country = (frame.name_c.to_numpy(dtype=object), frame.addr_c.to_numpy(dtype=object),
                                      frame.house.to_numpy(dtype=object), frame.country.to_numpy(dtype=object))
    for s in range(0, len(frame), chunk):
        ks, rs = [], []
        for i in range(s, min(s + chunk, len(frame))):
            k = _keys_for(name_c[i], addr_c[i], house[i], country[i])
            ks.extend(k)
            rs.extend([i] * len(k))
        hs.append(pd.util.hash_array(np.array(ks, dtype=object)).astype(np.int64))
        rows.append(np.asarray(rs, dtype=np.int32))
    return np.concatenate(hs), np.concatenate(rows)


def block(s1, other, cap=150, s1_chunk=200_000):
    """Return a DataFrame (i1, i2, n_keys) of candidate pairs.

    `s1` and `other` are prepped frames (positional row index = i1 / i2).
    """
    h2, r2 = key_table(other)
    # drop over-populated keys
    uniq, inv, cnt = np.unique(h2, return_inverse=True, return_counts=True)
    keep = cnt[inv] <= cap
    log(f"blocking: {len(h2):,} S2/S3 keys, {len(uniq):,} distinct, "
        f"{(cnt > cap).sum():,} keys over cap={cap} dropped ({(~keep).sum():,} rows)")
    h2, r2 = h2[keep], r2[keep]
    order = np.argsort(h2, kind="stable")
    h2, r2 = h2[order], r2[order]

    h1, r1 = key_table(s1)
    parts = []
    for s in range(0, len(h1), s1_chunk * 6):
        hh, rr = h1[s:s + s1_chunk * 6], r1[s:s + s1_chunk * 6]
        lo = np.searchsorted(h2, hh, side="left")
        hi = np.searchsorted(h2, hh, side="right")
        n = hi - lo
        has = n > 0
        if not has.any():
            continue
        lo, n, rr = lo[has], n[has], rr[has]
        # expand ranges: for each S1 key, the S2/S3 rows sharing it
        rep_i1 = np.repeat(rr, n)
        offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        idx2 = r2[np.repeat(lo, n) + offsets]
        parts.append(np.stack([rep_i1, idx2], axis=1))
    if not parts:
        return pd.DataFrame({"i1": np.array([], np.int32), "i2": np.array([], np.int32), "n_keys": np.array([], np.int8)})
    pairs = np.concatenate(parts)
    code = pairs[:, 0].astype(np.int64) * (1 << 32) + pairs[:, 1].astype(np.int64)
    uniq, cnt = np.unique(code, return_counts=True)
    out = pd.DataFrame({"i1": (uniq >> 32).astype(np.int32), "i2": (uniq & 0xFFFFFFFF).astype(np.int32),
                        "n_keys": np.minimum(cnt, 127).astype(np.int8)})
    log(f"blocking: {len(out):,} candidate pairs for {len(s1):,} S1 ({len(out) / max(len(s1), 1):.1f} per S1); "
        f"S1 with >=1 candidate: {out.i1.nunique():,}")
    return out
