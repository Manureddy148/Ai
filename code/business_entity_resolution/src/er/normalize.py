"""Text normalisation for business names and addresses.

All functions are pure and operate on single strings; `prep()` applies them to a whole
source frame with a process pool because the corpus has ~10M records per split.
"""
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from unidecode import unidecode

NAME_ABBR = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "intl": "international", "mfg": "manufacturing", "svc": "services",
    "svcs": "services", "bros": "brothers", "assoc": "associates", "natl": "national",
    "tech": "technologies", "grp": "group", "ent": "enterprises", "cie": "compagnie",
    "ste": "societe", "llp": "llp",
}
LEGAL = {
    "corporation", "company", "incorporated", "limited", "private", "llc", "llp", "lp",
    "plc", "pllc", "the", "and", "of", "dba", "pvt", "opc", "inc", "ltd", "co",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "societe", "compagnie", "et",
}
TLD = {"com", "in", "net", "org", "co", "io", "fr", "us", "biz", "info"}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "pkwy": "parkway", "ct": "court", "pl": "place", "sq": "square", "ste": "suite",
    "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "nr": "near", "opp": "opposite", "ngr": "nagar", "clny": "colony", "sec": "sector",
    "mkt": "market", "stn": "station", "marg": "road", "salai": "road",
    "fbg": "faubourg", "chem": "chemin", "imp": "impasse",
}
ADDR_STOP = {"near", "opposite", "behind", "beside", "next", "to", "the", "of", "and",
             "no", "number", "de", "la", "le", "du", "des", "null", "none", "nan", "n", "a"}

_punct = re.compile(r"[^a-z0-9 ]+")
_space = re.compile(r"\s+")
_digits = re.compile(r"\d+")
_postal = re.compile(r"\b(\d{5,6})\b")
_dotted = re.compile(r"(?<![a-z])(?:[a-z]\.){2,}[a-z]?(?![a-z])")
_ALIAS = re.compile(
    r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|trading as|t\s*/\s*a"
    r"|a\s*/\s*k\s*/\s*a|a\.k\.a\.?|also known as|formerly(?: known as)?)\b", re.I)


def basic(s):
    """ASCII-fold (incl. Indic scripts via unidecode), lowercase, strip punctuation."""
    s = unidecode(str(s)).lower()
    s = s.replace("&", " and ").replace("@", " at ")
    s = _dotted.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _punct.sub(" ", s)
    return _space.sub(" ", s).strip()


def norm_name(raw):
    """Normalised name with legal suffixes and web TLDs removed ('core name')."""
    s = basic(" ".join(_ALIAS.split(str(raw))))
    toks = [NAME_ABBR.get(t, t) for t in s.split()]
    if len(toks) > 1 and toks[-1] in TLD:          # 'maurewilliams com' -> 'maurewilliams'
        toks = toks[:-1]
    core = [t for t in toks if t not in LEGAL]
    return " ".join(core) if core else " ".join(toks)


def norm_addr(raw):
    """Normalised address with abbreviations expanded and filler tokens removed.

    Returns (addr_c, postal, house, nums) where postal is the 5-6 digit code (last one
    in the raw string), house the first token containing a digit, and nums the
    space-joined digit runs excluding the postal code.
    """
    postal = (_postal.findall(str(raw)) or [""])[-1]
    s = basic(raw)
    toks = [ADDR_ABBR.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in ADDR_STOP]
    house = next((t for t in toks if any(c.isdigit() for c in t)), "")
    nums = _digits.findall(s)
    if postal and postal in nums:
        nums.remove(postal)
    return " ".join(toks), postal, house, " ".join(dict.fromkeys(nums))


def _prep_chunk(args):
    names, addrs = args
    out_name = [norm_name(n) for n in names]
    out_addr = [norm_addr(a) for a in addrs]
    return out_name, out_addr


def prep(df, workers=4, chunk=50_000):
    """Add normalised columns to a source frame (entity_id, business_name, business_address, country)."""
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    name_c, addr_rows = [], []
    with Pool(workers) as pool:
        for n, a in pool.imap(_prep_chunk, jobs):
            name_c.extend(n)
            addr_rows.extend(a)
    out = pd.DataFrame({
        "entity_id": df.entity_id.to_numpy(),
        "country": pd.Series(df.country.map(basic).to_numpy(), dtype="string[pyarrow]"),
        "name_c": pd.Series(name_c, dtype="string[pyarrow]"),
        "addr_c": pd.Series([r[0] for r in addr_rows], dtype="string[pyarrow]"),
        "postal": pd.Series([r[1] for r in addr_rows], dtype="string[pyarrow]"),
        "house": pd.Series([r[2] for r in addr_rows], dtype="string[pyarrow]"),
        "nums": pd.Series([r[3] for r in addr_rows], dtype="string[pyarrow]"),
        "non_ascii": df.business_name.str.contains(r"[^\x00-\x7F]", regex=True).to_numpy().astype(np.int8),
        "src": df.entity_id.str[:2].to_numpy(),
    })
    return out
