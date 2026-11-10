import csv, json, math, os, subprocess, sys, time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
RM = Path(os.environ.get("BASE_ROOT", "."))
VF = Path(os.environ.get("DOCKING_ROOT", "."))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/utils")); sys.path.insert(0, str(HERE))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/charge_state.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("CHARGE_OUT_ROOT", str(ROOT)))
RES, RAW = OUT_ROOT / "results/charge_state", OUT_ROOT / "raw_outputs/charge_state"
PH = CFG["stats"]["ph"]
MISSING = -999
Z = NormalDist().inv_cdf(1 - CFG["stats"]["alpha"] / 2)


def E(s):
    return Path(s.replace("$BASE_ROOT", str(RM)).replace("$DOCKING_ROOT", str(VF)).replace("$ANALYSIS_ROOT", str(ROOT)))


PATHS = {k: E(v) for k, v in CFG["paths"].items()}


def check_pins():
    bad = [k for k, m in CFG["pinned_md5"].items() if P.file_md5(PATHS[k]) != m]
    if bad:
        sys.exit("pinned input changed: %s" % bad)


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def wilson(k, n, z=Z):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


import dimorphite_dl
import dominant_state as O

TABLE = O.load_pka_table(str(PATHS["dimorphite_smarts"]))


def analyze_one(smi):
    r = O.analyze_compound(smi, TABLE)
    if r is None:
        return None
    return dict(single_charge=r["single_charge"], single_smiles=r["single_smiles"], dom_charge=r["dom_charge"], dom_smiles=r["dom_smiles"],
                pop_smiles=r["pop_argmax_smiles"], pop_charge=O.fcharge(r["pop_argmax_smiles"]), pop_dom=r["pop_dom"],
                dom_is_popmax=r["dom_is_popmax"], n_sites=r["n_sites"], site_pkas=r["site_pkas"], n_microstates=r["n_microstates"])


def dimorphite_set_charges(smi):
    try:
        states = dimorphite_dl.protonate_smiles(smi, ph_min=PH, ph_max=PH)
    except Exception:
        return None
    out = set()
    for s in states:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.add(Chem.GetFormalCharge(m))
    return out or None


def openbabel_batch(smiles):
    inp = "\n".join("%s\t%d" % (s, i) for i, s in enumerate(smiles)) + "\n"
    p = subprocess.run([str(PATHS["obabel"]), "-ismi", "-osmi", "-p", str(PH)], input=inp, capture_output=True, text=True)
    out = {}
    for line in p.stdout.strip().splitlines():
        a = line.split("\t")
        if len(a) >= 2:
            m = Chem.MolFromSmiles(a[0].strip())
            ok = m is not None
            if m is None:
                m = Chem.MolFromSmiles(a[0].strip(), sanitize=False)
            if m is not None:
                out[int(a[1])] = (Chem.GetFormalCharge(m), a[0].strip(), ok)
    return out


def has_divalent_o_anion(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    return m is not None and any(a.GetSymbol() == "O" and a.GetFormalCharge() == -1 and a.GetDegree() == 2 for a in m.GetAtoms())


def run_ml(smiles, tag):
    RAW.mkdir(parents=True, exist_ok=True)
    uniq = list(dict.fromkeys(smiles))
    fin, fout = RAW / ("ml_in_%s.tsv" % tag), RAW / ("ml_sites_%s.jsonl" % tag)
    with open(fin, "w", encoding="utf-8") as f:
        f.write("smiles\n" + "\n".join(uniq) + "\n")
    p = subprocess.run([str(PATHS["ml_python"]), "-B", str(PATHS["ml_wrapper"]), str(fin), str(fout), str(CFG["stats"]["threads"])],
                       capture_output=True, text=True, cwd=str(ROOT))
    log("ml", tag, p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "", p.stderr.strip()[-200:])
    if p.returncode != 0:
        sys.exit("MolGpKa wrapper failed: " + p.stderr[-400:])
    return {r["smiles"]: r for r in (json.loads(l) for l in open(fout, encoding="utf-8"))}


def ml_state(rec, ph=PH):
    if rec is None or "error" in rec:
        return None
    nb = sum(1 for pk in rec["base"] if pk > ph)
    na = sum(1 for pk in rec["acid"] if pk < ph)
    pop = 1.0
    for pk in rec["base"]:
        pp = 1.0 / (1.0 + 10.0 ** (ph - pk)); pop *= max(pp, 1 - pp)
    for pk in rec["acid"]:
        pp = 1.0 / (1.0 + 10.0 ** (pk - ph)); pop *= max(pp, 1 - pp)
    return rec["q0"] + nb - na, nb, na, pop


CLASSES = CFG["agreement"]["charge_classes"]


def parse_any(smi):
    m = Chem.MolFromSmiles(smi)
    return m if m is not None else Chem.MolFromSmiles(smi, sanitize=False)


def state_class(smi):
    m = parse_any(smi)
    if m is None:
        return None
    q = Chem.GetFormalCharge(m)
    pos = neg = 0
    for a in m.GetAtoms():
        c = a.GetFormalCharge()
        if c == 0:
            continue
        if any(n.GetFormalCharge() * c < 0 for n in a.GetNeighbors()):
            continue
        pos += c > 0; neg += c < 0
    return charge_class(q, pos, neg)


def charge_class(q, free_pos, free_neg):
    if q == 0 and free_pos > 0 and free_neg > 0:
        return "zwitterion"
    return "<=-2" if q <= -2 else "-1" if q == -1 else "0" if q == 0 else "+1" if q == 1 else ">=+2"


def ml_class(q0, nb, na):
    return charge_class(q0 + nb - na, nb + max(q0, 0), na + max(-q0, 0))


CLASS_SMARTS = [
    ("amidine_guanidine", "[NX3;!$(N-C=[O,S]);!$(N-S(=O)=O)][CX3;!$(C-[OX2,SX2])]=[NX2;!$(N-[O,S,N]);!$(N-C=O);!$(N-S(=O)=O);!$(N-C#N)]", 1),
    ("aliphatic_amine", "[NX3;+0;!$(N~[!#6;!#1]);!$(N-[#6;!X4])]", 0),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 0),
    ("tetrazole_acylsulfonamide", "c1nn[nH]n1", 3),
    ("tetrazole_acylsulfonamide", "c1nnn[nH]1", 4),
    ("tetrazole_acylsulfonamide", "[#6](=O)[NX3H1]S(=O)(=O)", 2),
    ("imidazole_azole", "[nX2;r5;$(n:c:[nX3;r5])]", 0),
    ("pyridine_azine", "[nX2;+0]", 0),
    ("aniline", "[NX3;+0;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$(N-[#7,#8])]-a", 0),
    ("phenol", "[OX2H1]-a", 0),
    ("sulfonamide_imide", "[NX3H1,NX3H2;!$(N-C=O)]S(=O)(=O)", 0),
    ("sulfonamide_imide", "[#6](=O)[NX3H1][#6](=O)", 2),
]
CLASS_PATTERNS = [(c, Chem.MolFromSmarts(s), k) for c, s, k in CLASS_SMARTS]
assert all(p is not None for _, p, _ in CLASS_PATTERNS)
BASE_CLASSES = {"amidine_guanidine", "aliphatic_amine", "imidazole_azole", "pyridine_azine", "aniline"}
PRIORITY = CFG["classes"]["priority"]
NOMINAL = CFG["classes"]["nominal_pka_window"]


def _ring_ok(mol, a, cname):
    if cname not in ("imidazole_azole", "pyridine_azine"):
        return True
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if a not in ring or len(ring) != 5:
            continue
        het = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
        nn = [i for i in het if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
        if len(nn) >= 4:
            return False
        if cname == "imidazole_azole":
            adj = len(nn) == 2 and mol.GetBondBetweenAtoms(nn[0], nn[1]) is not None
            return len(het) == 2 and len(nn) == 2 and not adj
    return cname != "imidazole_azole" or False


def independent_sites(mol):
    seen, out = set(), []
    for cname in PRIORITY:
        for c, patt, k in CLASS_PATTERNS:
            if c != cname:
                continue
            for match in mol.GetSubstructMatches(patt):
                a = match[k]
                if a in seen or not _ring_ok(mol, a, c):
                    continue
                seen.add(a)
                lo, hi = NOMINAL[c]
                out.append((c, "base" if c in BASE_CLASSES else "acid", a, lo, hi))
    return out


def primary_class(sites):
    have = {s[0] for s in sites}
    for c in PRIORITY:
        if c in have:
            return c
    return "none"


def group_atoms(mol, atoms, kinds=None):
    atoms = list(atoms)
    parent = {a: a for a in atoms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nbr = {a: {n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 6} for a in atoms}
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            if (mol.GetAtomWithIdx(a).GetAtomicNum() == 7 and mol.GetAtomWithIdx(b).GetAtomicNum() == 7 and nbr[a] & nbr[b]
                    and (kinds is None or kinds[a] == kinds[b])):
                parent[find(a)] = find(b)
    return {a: find(a) for a in atoms}


def site_counts(mol, smi):
    a = CFG["anchor"]
    glo, ghi = a["gate_window"]; rlo, rhi = PH - a["relevant_halfwidth"], PH + a["relevant_halfwidth"]
    dd = O.detect_site_pkas(smi, TABLE)
    ind = independent_sites(mol)
    ov = lambda lo, hi, x, y: lo <= y and hi >= x

    def ngroups(items, kinds, x, y):
        sel = [at for at, lo, hi in items if ov(lo, hi, x, y)]
        return len(set(group_atoms(mol, sel, kinds).values())) if sel else 0

    dd_items = [(at, p, p) for at, p in dd]
    ind_items = [(s[2], s[3], s[4]) for s in ind]
    ind_kind = {s[2]: s[1] for s in ind}
    return dict(n_dd_strict=ngroups(dd_items, None, glo, ghi), n_dd_relaxed=ngroups(dd_items, None, rlo, rhi),
                n_ind_strict=ngroups(ind_items, ind_kind, glo, ghi), n_ind_relaxed=ngroups(ind_items, ind_kind, rlo, rhi),
                n_ind_base=sum(1 for s in ind if s[1] == "base"), n_ind_acid=sum(1 for s in ind if s[1] == "acid"),
                primary_class=primary_class(ind), classes=sorted({s[0] for s in ind}))


_UNCH = rdMolStandardize.Uncharger()


def clean_mol(m):
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    big = frags[max(range(len(frags)), key=lambda i: (frags[i].GetNumHeavyAtoms(), -i))]
    big = _UNCH.uncharge(big)
    return Chem.MolFromSmiles(Chem.MolToSmiles(big))


def skeleton_key(m):
    try:
        c = clean_mol(m)
        return Chem.MolToInchiKey(c)[:14] if c is not None else None
    except Exception:
        return None


def standardize(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return dict(reason="unparsable")
    big = clean_mol(m)
    if big is None:
        return dict(reason="unparsable_after_neutralisation")
    can = Chem.MolToSmiles(big)
    return dict(reason="", smiles=can, inchikey=Chem.MolToInchiKey(big), net_charge=Chem.GetFormalCharge(big),
                elements_ok=all(a.GetSymbol() in CFG["anchor"]["allowed_elements"] for a in big.GetAtoms()),
                mw=Descriptors.MolWt(big), rings=rdMolDescriptors.CalcNumRings(big), hbd=Lipinski.NumHDonors(big), hba=Lipinski.NumHAcceptors(big),
                heavy=big.GetNumHeavyAtoms())


def druglike(s):
    d = CFG["anchor"]["druglike"]
    return d["mw_min"] <= s["mw"] <= d["mw_max"] and s["rings"] >= d["min_rings"] and s["hbd"] <= d["hbd_max"] and s["hba"] <= d["hba_max"]


def hh_pmf(consts, ph=PH):
    pmf = {0: 1.0}
    for d, pk in consts:
        if d == "AH":
            p, s = 1.0 / (1.0 + 10.0 ** (ph - pk)), 1
        else:
            p, s = 1.0 / (1.0 + 10.0 ** (pk - ph)), -1
        new = {}
        for q, w in pmf.items():
            new[q] = new.get(q, 0.0) + w * (1 - p)
            new[q + s] = new.get(q + s, 0.0) + w * p
        pmf = new
    return pmf


def dominant_of(pmf):
    q = max(pmf, key=lambda k: (round(pmf[k], 12), -abs(k), -k))
    return q, pmf[q]


def ladder_dominant(consts, ph=PH):
    if not consts:
        return 0, 1.0
    nb = sum(1 for d, _ in consts if d == "AH")
    pks = sorted(pk for _, pk in consts)
    logw, acc = [0.0], 0.0
    for pk in pks:
        acc += ph - pk; logw.append(acc)
    mx = max(logw); w = [10.0 ** (x - mx) for x in logw]; t = sum(w)
    j = max(range(len(w)), key=lambda i: w[i])
    return nb - j, w[j] / t


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return format(float(v), ".10g")
    return str(v)


def write_csv(path, cols, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = [c[0] for c in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(names)
        for r in rows:
            extra = set(r) - set(names)
            if extra:
                raise ValueError("%s: unexpected keys %s" % (path.name, sorted(extra)))
            w.writerow([fmt(r.get(n)) for n in names])
