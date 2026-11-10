import sys, time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "$ANALYSIS_ROOT/scripts/charge_state")
import decoy_pipeline as PL
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import charge_common as S3

_UNC = rdMolStandardize.Uncharger()


def dominant_charge(std):
    from dimorphite_dl import protonate_smiles
    out = protonate_smiles(std, ph_min=7.4, ph_max=7.4, precision=0.0)
    for s in out:
        m = Chem.MolFromSmiles(s.split()[0])
        if m is not None:
            return Chem.GetFormalCharge(m)
    return None


def n_sites_of(std):
    m = Chem.MolFromSmiles(std)
    neutral = _UNC.uncharge(m)
    sites = S3.independent_sites(neutral)
    if not sites:
        return 0
    kinds = {s[2]: s[1] for s in sites}
    return len(set(S3.group_atoms(neutral, [s[2] for s in sites], kinds).values()))


def cheap_props(smiles):
    t0 = time.time()
    try:
        std = PL.ingest(smiles)
        return dict(ok=True, std_smiles=std, dominant_charge=dominant_charge(std), error="", seconds=time.time() - t0)
    except Exception as exc:
        return dict(ok=False, std_smiles="", dominant_charge=None, error=str(exc)[:80], seconds=time.time() - t0)


def full_props(smiles):
    t0 = time.time()
    r = dict(ok=False, std_smiles="", error="", dominant_charge=None, n_sites=None, n_microstates_all=None, n_selected=None, n_microstates=None, n_tautomers=None, min_q=None, max_q=None,
             n_rot=None, n_heavy=None, mw=None, seconds=None)
    try:
        std = PL.ingest(smiles)
        r["std_smiles"] = std
        m = Chem.MolFromSmiles(std)
        r["n_rot"] = int(rdMolDescriptors.CalcNumRotatableBonds(m)); r["n_heavy"] = int(m.GetNumHeavyAtoms()); r["mw"] = float(Descriptors.MolWt(m))
        r["dominant_charge"] = dominant_charge(std)
        r["n_sites"] = n_sites_of(std)
        e = PL.enumerate_ligand(smiles, std=std)
        r["n_microstates_all"] = len(e["all"]); r["n_selected"] = len(e["selected"]); r["n_microstates"] = len(e["eligible"])
        r["n_tautomers"] = len({x["tautomer_key"] for x in e["eligible"]})
        qs = [x["formal_charge"] for x in e["eligible"]]
        r["min_q"], r["max_q"] = (min(qs), max(qs)) if qs else (None, None)
        r["ok"] = r["n_microstates"] > 0
    except Exception as exc:
        r["error"] = str(exc)[:80]
    r["seconds"] = time.time() - t0
    return r
