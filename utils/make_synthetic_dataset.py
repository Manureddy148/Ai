"""Synthetic dataset in the challenge's exact format, for smoke-testing er_pipeline.py end-to-end.

Usage: python3 utils/make_synthetic_dataset.py <out_dir>
Writes <out_dir>/train/{train_source1,2,3,train_ground_truth}.tsv, <out_dir>/test/test_source{1,2,3}.tsv
and <out_dir>/test_ground_truth_hidden.tsv (kept outside test/ like the real hidden labels) with US, India
(train+test) and France (test only), noisy S2/S3 variants of S1 records, singletons and distractors.
"""
import os
import random
import sys

random.seed(7)
OUT = sys.argv[1] if len(sys.argv) > 1 else "synth"

WORDS = ["Global", "Sunrise", "Apex", "Blue", "Harbor", "Krishna", "Lakshmi", "Metro", "Pioneer",
         "Royal", "Silver", "Star", "United", "Vertex", "Zenith", "Delta", "Omega", "Nova", "Alpha",
         "Bharat", "Ganga", "Shree", "Om", "Sai", "Lumière", "Château", "Provence", "Étoile", "Rivière"]
KIND = ["Traders", "Textiles", "Electronics", "Logistics", "Foods", "Motors", "Pharma", "Consulting",
        "Boulangerie", "Immobilier", "Software", "Engineering", "Hardware", "Bakery", "Studio"]
SUFFIX = {"US": ["Inc", "LLC", "Corp", "Co.", ""], "India": ["Pvt Ltd", "Private Limited", "& Sons", "Enterprises", ""],
          "France": ["SARL", "SAS", "SA", "EURL", ""]}
STREET = {"US": ["Main St", "Oak Avenue", "5th Ave", "Elm Rd", "Park Blvd", "Lake Dr"],
          "India": ["MG Road", "Nehru Nagar", "Anna Salai", "Brigade Rd", "Sector 17", "Station Road"],
          "France": ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Saint-Germain", "Chemin des Vignes"]}
CITY = {"US": [("Austin", "TX", "78701"), ("Denver", "CO", "80202"), ("Boston", "MA", "02110")],
        "India": [("Bengaluru", "Karnataka", "560001"), ("Chennai", "Tamil Nadu", "600002"), ("Pune", "Maharashtra", "411001")],
        "France": [("Paris", "", "75008"), ("Lyon", "", "69002"), ("Marseille", "", "13001")]}
ABBR = {"Street": "St", "St": "Street", "Road": "Rd", "Rd": "Road", "Avenue": "Ave", "Ave": "Avenue",
        "Private Limited": "Pvt Ltd", "Pvt Ltd": "Private Limited", "Corp": "Corporation",
        "Inc": "Incorporated", "&": "and", "and": "&", "Boulevard": "Blvd", "Blvd": "Boulevard"}


def typo(s):
    if len(s) < 6:
        return s
    i = random.randrange(1, len(s) - 1)
    ops = [s[:i] + s[i + 1:], s[:i] + s[i + 1] + s[i] + s[i + 2:], s[:i] + random.choice("aeiou") + s[i:]]
    return random.choice(ops)


def gen_entity(country):
    name = f"{random.choice(WORDS)} {random.choice(KIND)} {random.choice(SUFFIX[country])}".strip()
    city, state, pin = random.choice(CITY[country])
    num = random.randint(1, 999)
    addr = f"{num} {random.choice(STREET[country])}, {city}{', ' + state if state else ''} {pin}"
    return name, addr


def perturb(name, addr, country):
    n, a = name, addr
    r = random.random()
    if r < 0.3:
        for k, v in ABBR.items():
            if k in n:
                n = n.replace(k, v, 1)
                break
    elif r < 0.5:
        n = typo(n)
    elif r < 0.6:
        n = n.upper()
    elif r < 0.7 and " " in n:
        toks = n.split()
        n = " ".join(toks[1:] + toks[:1])
    elif r < 0.8:
        n = n + " (DBA " + random.choice(WORDS) + " " + random.choice(KIND) + ")"
    r = random.random()
    if r < 0.3:
        for k, v in ABBR.items():
            if k in a:
                a = a.replace(k, v, 1)
                break
    elif r < 0.5:
        a = a.rsplit(" ", 1)[0]          # drop PIN
    elif r < 0.65:
        a = "Near " + random.choice(["SBI ATM", "Bus Stand", "City Mall", "Temple"]) + ", " + a
    elif r < 0.8:
        a = typo(a)
    elif r < 0.9:
        parts = a.split(", ")
        a = ", ".join(reversed(parts))
    return n, a


def build(split, n_s1, countries):
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 0
    for i in range(n_s1):
        country = random.choice(countries)
        name, addr = gen_entity(country)
        sid = f"S1-{split[0].upper()}{i:05d}"
        s1.append((sid, name, addr, country))
        matches = []
        k = random.choices([0, 1, 2, 3], weights=[0.35, 0.4, 0.18, 0.07])[0]
        for _ in range(k):
            n, a = perturb(name, addr, country)
            if random.random() < 0.5:
                c2 += 1; eid = f"S2-{split[0].upper()}{c2:05d}"; s2.append((eid, n, a, country))
            else:
                c3 += 1; eid = f"S3-{split[0].upper()}{c3:05d}"; s3.append((eid, n, a, country))
            matches.append(eid)
        gt.append((sid, ",".join(matches)))
    for _ in range(n_s1 // 2):        # distractors
        country = random.choice(countries)
        n, a = gen_entity(country)
        if random.random() < 0.5:
            c2 += 1; s2.append((f"S2-{split[0].upper()}{c2:05d}", n, a, country))
        else:
            c3 += 1; s3.append((f"S3-{split[0].upper()}{c3:05d}", n, a, country))
    random.shuffle(s2); random.shuffle(s3)
    d = os.path.join(OUT, split)
    os.makedirs(d, exist_ok=True)
    for tag, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
        with open(os.path.join(d, f"{split}_{tag}.tsv"), "w") as fh:
            fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")
    gt_path = os.path.join(d, "train_ground_truth.tsv") if split == "train" \
        else os.path.join(OUT, "test_ground_truth_hidden.tsv")   # kept outside test/ like the real hidden labels
    if True:
        with open(gt_path, "w") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            for r in gt:
                fh.write("\t".join(r) + "\n")
    print(split, "S1", len(s1), "S2", len(s2), "S3", len(s3))


build("train", 3000, ["US", "India"])
build("test", 1500, ["US", "India", "France"])
