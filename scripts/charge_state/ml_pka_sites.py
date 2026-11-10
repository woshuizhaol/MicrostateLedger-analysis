import csv, json, os, sys, warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
M = "$ANALYSIS_ROOT/inputs/external/ml_pka_tools/MolGpKa"
os.chdir(M + "/src")
sys.path.insert(0, os.getcwd())
import torch
torch.set_num_threads(1)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from utils.ionization_group import get_ionization_aid
from predict_pka import load_model, model_pred

MODELS = {}


def models():
    if not MODELS:
        MODELS["acid"] = load_model(M + "/models/weight_acid.pth")
        MODELS["base"] = load_model(M + "/models/weight_base.pth")
    return MODELS


def heavy_atom(mol, a):
    at = mol.GetAtomWithIdx(a)
    return at.GetNeighbors()[0].GetIdx() if at.GetAtomicNum() == 1 else a


def groups(mol, per_atom, kind):
    atoms = sorted(per_atom)
    parent = {a: a for a in atoms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nbr = {a: {n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 6} for a in atoms}
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            if mol.GetAtomWithIdx(a).GetAtomicNum() == 7 and mol.GetAtomWithIdx(b).GetAtomicNum() == 7 and nbr[a] & nbr[b]:
                parent[find(a)] = find(b)
    out = {}
    for a in atoms:
        v = min(per_atom[a]) if kind == "acid" else max(per_atom[a])
        r = find(a)
        out[r] = v if r not in out else (min(out[r], v) if kind == "acid" else max(out[r], v))
    return sorted(out.values())


def one(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return dict(smiles=smi, error="unparsable")
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        q0 = Chem.GetFormalCharge(mol)
        mol = AllChem.AddHs(mol)
        out = dict(smiles=smi, q0=q0)
        for kind in ("base", "acid"):
            per_atom = {}
            raw = 0
            for a in get_ionization_aid(mol, acid_or_base=kind):
                per_atom.setdefault(heavy_atom(mol, a), []).append(float(model_pred(mol, a, models()[kind])))
                raw += 1
            out[kind] = groups(mol, per_atom, kind)
            out["n_raw_" + kind] = raw
        return out
    except Exception as e:
        return dict(smiles=smi, error=type(e).__name__ + ": " + str(e)[:120])


if __name__ == "__main__":
    inp, outp, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
    smis = [r["smiles"] for r in csv.DictReader(open(inp, encoding="utf-8"), delimiter="\t")]
    uniq = list(dict.fromkeys(smis))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(nproc) as pool:
        res = pool.map(one, uniq, chunksize=8)
    with open(outp, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print("molecules", len(uniq), "errors", sum(1 for r in res if "error" in r))
