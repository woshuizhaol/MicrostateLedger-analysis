import argparse, glob, hashlib, json, os, random, subprocess, sys, tempfile, traceback
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from dimorphite_dl.protonate.run import protonate_smiles

CORE = "$DOCKING_ROOT/data/processed/CASF-2016/CASF-2016/coreset"
CORESET_DAT = "$DOCKING_ROOT/data/processed/CASF-2016/CASF-2016/power_scoring/CoreSet.dat"
OUT = os.environ.get("CASF_WORK", "$ANALYSIS_ROOT/raw_outputs/casf_scoring/scoring_work")
VINA = "/usr/bin/vina"
SMINA = "$ANALYSIS_ROOT/env/smina_link/bin/smina"
MK_LIG = "mk_prepare_ligand.py"
SEED = 20260704
MAX_ENUM = 24

os.makedirs(f"{OUT}/receptors", exist_ok=True)
os.makedirs(f"{OUT}/receptors_q", exist_ok=True)
os.environ["PATH"] = "$ANALYSIS_ROOT/env/conda/bin:" + os.environ["PATH"]

def load_native(pdbid, wd):
    d = f"{CORE}/{pdbid}"
    mol2 = f"{d}/{pdbid}_ligand.mol2"; sdf = f"{d}/{pdbid}_ligand.sdf"
    m = Chem.MolFromMol2File(mol2, removeHs=False, sanitize=True)
    method = "mol2_rdkit"
    if m is None:
        try: m = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True)[0]; method = "sdf_rdkit"
        except Exception: m = None
    if m is None:
        tmp = f"{wd}/native_fix.sdf"
        subprocess.run(["obabel", mol2, "-O", tmp], capture_output=True, text=True)
        try: m = Chem.SDMolSupplier(tmp, removeHs=False, sanitize=True)[0]; method = "mol2_obabel_sdf"
        except Exception: m = None
    if m is None:
        return None, None, None
    native_noH = Chem.RemoveHs(m)
    ob = subprocess.run(["obabel", mol2, "-osmi"], capture_output=True, text=True)
    smi_ob = ob.stdout.strip().split()[0] if ob.stdout.strip() else None
    smi = smi_ob if (smi_ob and Chem.MolFromSmiles(smi_ob) is not None) else Chem.MolToSmiles(native_noH)
    return native_noH, smi, method

def neutralize_smiles(native_noH):
    try:
        un = rdMolStandardize.Uncharger()
        m = un.uncharge(Chem.Mol(native_noH))
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        return None

def enumerate_states(smi, native_noH):
    def canon(s):
        m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else None
    def q(s):
        m = Chem.MolFromSmiles(s); return Chem.GetFormalCharge(m) if m else None
    ens = [canon(s) for s in protonate_smiles(smi, ph_min=6.4, ph_max=8.4, precision=1.0)]
    ens = [s for s in ens if s]
    ens = list(dict.fromkeys(ens))
    truncated = False
    if len(ens) > MAX_ENUM:
        ens = sorted(ens, key=lambda s: (abs(q(s)), s))[:MAX_ENUM]; truncated = True
    dom = protonate_smiles(smi, ph_min=7.4, ph_max=7.4, precision=0.0)
    dominant = canon(dom[0]) if dom else None
    single = sorted(ens, key=lambda s: (abs(q(s)), s))[0] if ens else None
    neutral = neutralize_smiles(native_noH)
    inp = canon(smi)
    stable = int(hashlib.sha256((inp or "").encode()).hexdigest()[:8], 16)
    rng = random.Random(SEED + stable)
    non_default = [s for s in ens if s != single]
    rand = rng.choice(non_default) if non_default else None
    states = {"input": inp, "neutral": neutral, "single": single, "dominant": dominant, "random": rand}
    return states, ens, truncated

def neutral_skeleton(mol):
    m = Chem.RWMol(Chem.RemoveHs(Chem.Mol(mol)))
    for a in m.GetAtoms():
        a.SetFormalCharge(0); a.SetNoImplicit(True); a.SetNumExplicitHs(0); a.SetIsAromatic(False)
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE); b.SetIsAromatic(False)
    mm = m.GetMol(); Chem.FastFindRings(mm); return mm

def map_to_native(ms_smi, native_noH, sk_nat):
    ms = Chem.MolFromSmiles(ms_smi)
    if ms is None: return None
    ms_noH = Chem.RemoveHs(ms)
    match = sk_nat.GetSubstructMatch(neutral_skeleton(ms_noH))
    if len(match) != ms_noH.GetNumAtoms(): return None
    conf = Chem.Conformer(ms_noH.GetNumAtoms()); nc = native_noH.GetConformer()
    for i, nat in enumerate(match):
        conf.SetAtomPosition(i, nc.GetAtomPosition(nat))
    ms_noH.RemoveAllConformers(); ms_noH.AddConformer(conf, assignId=True)
    try: return Chem.AddHs(ms_noH, addCoords=True)
    except Exception: return None

def lig_pdbqt(msh, path_sdf, path_pdbqt):
    w = Chem.SDWriter(path_sdf); w.write(msh); w.close()
    r = subprocess.run([MK_LIG, "-i", path_sdf, "-o", path_pdbqt], capture_output=True, text=True)
    if os.path.exists(path_pdbqt) and os.path.getsize(path_pdbqt) > 0:
        return "meeko"
    subprocess.run(["obabel", path_sdf, "-O", path_pdbqt], capture_output=True, text=True)
    if os.path.exists(path_pdbqt) and os.path.getsize(path_pdbqt) > 0:
        return "obabel"
    return None

def receptor_pdbqt(pdbid):
    out = f"{OUT}/receptors/{pdbid}.pdbqt"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    prot = f"{CORE}/{pdbid}/{pdbid}_protein.pdb"
    subprocess.run(["obabel", prot, "-xr", "-O", out], capture_output=True, text=True)
    return out if (os.path.exists(out) and os.path.getsize(out) > 0) else None

def box_from(native_noH):
    c = native_noH.GetConformer()
    xyz = np.array([list(c.GetAtomPosition(i)) for i in range(native_noH.GetNumAtoms())])
    cen = xyz.mean(0); ext = xyz.max(0) - xyz.min(0)
    size = [float(max(22.5, e + 10)) for e in ext]
    return cen, size

def vina_score(receptor, ligpdbqt, cen, size):
    cmd = [VINA, "--receptor", receptor, "--ligand", ligpdbqt, "--score_only",
           "--center_x", f"{cen[0]:.3f}", "--center_y", f"{cen[1]:.3f}", "--center_z", f"{cen[2]:.3f}",
           "--size_x", f"{size[0]:.1f}", "--size_y", f"{size[1]:.1f}", "--size_z", f"{size[2]:.1f}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        s = ln.strip()
        if s.startswith("Estimated Free Energy of Binding") or s.startswith("Affinity"):
            for tok in s.replace(":", " ").split():
                try: return float(tok)
                except ValueError: continue
    return None

def smina_score(receptor, ligpdbqt):
    r = subprocess.run([SMINA, "--score_only", "-r", receptor, "-l", ligpdbqt], capture_output=True, text=True)
    aff = [float(ln.split()[1]) for ln in r.stdout.splitlines() if ln.startswith("Affinity:")]
    return aff[0] if aff else None

def fcharge(smi):
    m = Chem.MolFromSmiles(smi); return Chem.GetFormalCharge(m) if m else None



def _ok(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

def receptor_q(pdbid):
    out = f"{OUT}/receptors_q/{pdbid}.pdbqt"
    mth = f"{OUT}/receptors_q/{pdbid}.method"
    if _ok(out):
        return out, (open(mth).read().strip() if os.path.exists(mth) else "cached")
    mol2 = f"{CORE}/{pdbid}/{pdbid}_protein.mol2"
    pdb = f"{CORE}/{pdbid}/{pdbid}_protein.pdb"
    attempts = [
        ("mol2_gasteiger", ["obabel", mol2, "-xr", "--partialcharge", "gasteiger", "-O", out]),
        ("pdb_gasteiger", ["obabel", pdb, "-xr", "--partialcharge", "gasteiger", "-O", out]),
        ("mol2_native_charge", ["obabel", mol2, "-xr", "-O", out]),
        ("pdb_zero", ["obabel", pdb, "-xr", "-O", out]),
    ]
    for name, cmd in attempts:
        if os.path.exists(out): os.remove(out)
        subprocess.run(cmd, capture_output=True, text=True)
        if _ok(out):
            nz = sum(1 for ln in open(out) if ln.startswith(("ATOM", "HETATM"))
                     and ln.strip().split()[-2] not in ("+0.000", "0.000", "-0.000"))
            if nz > 0 or name == "pdb_zero":
                open(mth, "w").write(name)
                return out, name
    return (None, "fail")

def ad4_score(receptor, ligpdbqt):
    r = subprocess.run([SMINA, "--scoring", "ad4_scoring", "--score_only", "-r", receptor, "-l", ligpdbqt],
                       capture_output=True, text=True)
    aff = [float(ln.split()[1]) for ln in r.stdout.splitlines() if ln.startswith("Affinity:")]
    return aff[0] if aff else None

