import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from dimorphite_dl.protonate.run import protonate_smiles

PH = 7.4
RT = 0.0019872041 * 298.15
DD_VER = "2.0.2"
METHOD = (f"dimorphite_dl-{DD_VER} SMARTS-mean pKa; independent-site Henderson-Hasselbalch @ pH{PH}; "
          "dominant=protonate(precision=0); ensemble=protonate(6.4-8.4,precision=1.0)")


def canonical(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def fcharge(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.GetFormalCharge(m) if m is not None else None


def neutralize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        if a.GetFormalCharge() != 0:
            a.SetFormalCharge(0)
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return canonical(smi)
    return Chem.MolToSmiles(rw)


def load_pka_table(path):
    table = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            name, smarts = parts[0], parts[1]
            rest = parts[2:]
            sites = []
            i = 0
            while i + 2 < len(rest) + 1 and i + 1 < len(rest):
                try:
                    idx = int(rest[i]); mean = float(rest[i + 1]); _ = float(rest[i + 2])
                    sites.append((idx, mean))
                    i += 3
                except (ValueError, IndexError):
                    break
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and sites:
                table.append((name, patt, sites))
    return table


def detect_site_pkas(neutral_smi, table):
    m = Chem.MolFromSmiles(neutral_smi)
    if m is None:
        return []
    mh = Chem.AddHs(m)
    seen_atoms = set()
    out = []
    for name, patt, sites in table:
        for match in mh.GetSubstructMatches(patt):
            for idx_site, mean in sites:
                if idx_site >= len(match):
                    continue
                atom = match[idx_site]
                if atom in seen_atoms or atom >= m.GetNumAtoms():
                    continue
                seen_atoms.add(atom)
                out.append((atom, mean))
    return out


def heavy_skeleton_order(mol):
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        a.SetNumRadicalElectrons(0)
    Chem.SanitizeMol(m, catchErrors=True)
    try:
        ranks = list(Chem.CanonicalRankAtoms(m, breakTies=True, includeChirality=False))
    except Exception:
        return None
    return sorted(range(len(ranks)), key=lambda i: ranks[i])


def member_site_charges(member_smi, parent_neutral_mol, site_atoms):
    mm = Chem.MolFromSmiles(member_smi)
    if mm is None:
        return None
    po = heavy_skeleton_order(parent_neutral_mol)
    mo = heavy_skeleton_order(mm)
    if po is None or mo is None or len(po) != len(mo):
        return None
    pos_of_parent = {pa: i for i, pa in enumerate(po)}
    out = {}
    for sa in site_atoms:
        i = pos_of_parent.get(sa)
        if i is None:
            return None
        out[sa] = mm.GetAtomWithIdx(mo[i]).GetFormalCharge()
    return out


def analyze_compound(parent_smi, table):
    cano = canonical(parent_smi)
    if cano is None:
        return None
    neutral = neutralize(parent_smi)
    if neutral is None:
        return None
    try:
        dom_list = list(protonate_smiles(parent_smi, ph_min=PH, ph_max=PH, precision=0.0))
        ens_list = list(protonate_smiles(parent_smi, ph_min=6.4, ph_max=8.4, precision=1.0))
    except Exception:
        return None
    if not dom_list or not ens_list:
        return None
    dom_smi = canonical(dom_list[0])
    q_dom = fcharge(dom_list[0])
    if dom_smi is None or q_dom is None:
        return None

    ens = {}
    for s in ens_list:
        c = canonical(s)
        if c is None:
            continue
        ens[c] = fcharge(c)
    if dom_smi not in ens:
        ens[dom_smi] = q_dom

    sites = detect_site_pkas(neutral, table)
    site_atoms = [a for a, _ in sites]
    site_pkas = [p for _, p in sites]
    n_sites = len(sites)
    p_proton = {a: 1.0 / (1.0 + 10.0 ** (PH - pka)) for a, pka in sites}
    parent_neutral_mol = Chem.MolFromSmiles(neutral)

    members = sorted(ens.keys(), key=lambda c: (abs(ens[c]), c))
    q_by = {c: ens[c] for c in members}

    member_sc = {}
    site_max_q = {a: None for a in site_atoms}
    for c in members:
        sc = member_site_charges(c, parent_neutral_mol, site_atoms) if site_atoms else {}
        member_sc[c] = sc
        if sc is None:
            continue
        for a, q in sc.items():
            if site_max_q[a] is None or q > site_max_q[a]:
                site_max_q[a] = q

    raw_pop = {}
    for c in members:
        sc = member_sc[c]
        if sc is None:
            raw_pop[c] = float("nan")
            continue
        prob = 1.0
        for a in site_atoms:
            pp = p_proton[a]
            prob *= pp if (site_max_q[a] is not None and sc[a] == site_max_q[a]) else (1.0 - pp)
        raw_pop[c] = prob
    tot = sum(v for v in raw_pop.values() if not math.isnan(v))
    pop = {c: (raw_pop[c] / tot if (tot > 0 and not math.isnan(raw_pop[c])) else float("nan"))
           for c in members}

    finite = {c: pop[c] for c in members if not math.isnan(pop[c])}
    pop_argmax = max(finite, key=finite.get) if finite else dom_smi
    pop_dom = pop.get(dom_smi, float("nan"))

    single_smi = min(members, key=lambda c: (abs(q_by[c]), c))

    rows = []
    for c in members:
        p = pop[c]
        fe = (-RT * math.log(p / pop_dom)) if (not math.isnan(p) and not math.isnan(pop_dom)
                                               and p > 0 and pop_dom > 0) else float("nan")
        pka_val = (site_pkas[0] if (n_sites == 1 and c != dom_smi) else float("nan"))
        rows.append({
            "smiles": c, "formal_charge": q_by[c], "population": p, "free_energy": fe,
            "pka": pka_val, "is_dominant": 1 if c == dom_smi else 0,
            "is_single": 1 if c == single_smi else 0,
        })
    return {
        "parent_smiles": cano, "neutral_smiles": neutral, "n_sites": n_sites,
        "site_pkas": site_pkas, "dom_smiles": dom_smi, "dom_charge": q_dom,
        "single_smiles": single_smi, "single_charge": q_by[single_smi],
        "pop_argmax_smiles": pop_argmax, "pop_dom": pop_dom,
        "mismatch_charge": int(q_dom != q_by[single_smi]),
        "mismatch_struct": int(dom_smi != single_smi),
        "dom_is_popmax": int(dom_smi == pop_argmax),
        "n_microstates": len(members), "rows": rows,
    }


SCHEMA = """
PRAGMA user_version = 2;
CREATE TABLE IF NOT EXISTS compounds (
  compound_id TEXT PRIMARY KEY, name TEXT, reg_hash TEXT, inchikey TEXT,
  created_at TEXT, parent_sdf_path TEXT, parent_smiles TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS microstates (
  microstate_id TEXT PRIMARY KEY, compound_id TEXT NOT NULL, fixedh_inchi TEXT,
  formal_charge INTEGER NOT NULL, stereo_tag TEXT, coordination_sig TEXT, created_at TEXT,
  sdf_path TEXT,
  population REAL, pka REAL, free_energy REAL, method TEXT, ph REAL, is_dominant INTEGER DEFAULT 0,
  FOREIGN KEY (compound_id) REFERENCES compounds(compound_id));
"""


