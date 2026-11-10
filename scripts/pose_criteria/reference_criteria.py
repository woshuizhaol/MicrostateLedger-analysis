import io, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
CORE = Path("$DOCKING_ROOT/data/processed/CASF-2016/CASF-2016/coreset")


def crystal_ligand(pdbid):
    m = Chem.MolFromMol2File(str(CORE / pdbid / ("%s_ligand.mol2" % pdbid)), removeHs=False, sanitize=True)
    return m


def first_mol(sdf_text_or_path):
    s = str(sdf_text_or_path)
    if "\n" in s:
        return next((x for x in Chem.ForwardSDMolSupplier(io.BytesIO(s.encode()), removeHs=False) if x is not None), None)
    return Chem.SDMolSupplier(s, removeHs=False)[0]


def rmsd_sym(crystal, pose):
    from spyrmsd import molecule, rmsd
    try:
        a = molecule.Molecule.from_rdkit(crystal); b = molecule.Molecule.from_rdkit(pose)
        a.strip(); b.strip()
        if a.coordinates.shape[0] != b.coordinates.shape[0]:
            return float("nan"), "heavy_atom_count_differs(%d vs %d)" % (a.coordinates.shape[0], b.coordinates.shape[0])
        r = rmsd.symmrmsd(a.coordinates, b.coordinates, a.atomicnums, b.atomicnums, a.adjacency_matrix, b.adjacency_matrix, center=False, minimize=False)
        return float(r), ""
    except Exception as exc:
        return float("nan"), "spyrmsd_error:" + str(exc)[:60]


_PROT = {}


ROUTE = {}
BACK = {}


def _receptor_block(pdbid):
    out, back, key_prev, n = [], {}, None, 0
    for l in open(CORE / pdbid / ("%s_protein.pdb" % pdbid)):
        if l.startswith("ATOM"):
            key = (l[21], l[22:27], l[17:20])
            if key != key_prev:
                n += 1; key_prev = key
                back["%s%d.%s" % (l[17:20].strip(), n, l[21].strip())] = "%s%s.%s" % (l[17:20].strip(), l[22:27].strip(), l[21].strip())
            l = l[:22] + "%4d" % n + " " + l[27:]
            out.append(l)
        elif l.startswith("TER"):
            out.append(l)
    return "".join(out), back


def protein(pdbid):
    import prolif as plf
    if pdbid not in _PROT:
        block, back = _receptor_block(pdbid)
        m = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=True)
        route = "rdkit"
        if m is None:
            m = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
            if m is None:
                raise ValueError("RDKit cannot read the receptor of " + pdbid)
            Chem.SanitizeMol(m, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            route = "rdkit_nopropcheck"
        _PROT[pdbid] = plf.Molecule(m); ROUTE[pdbid] = route; BACK[pdbid] = back
    return _PROT[pdbid]


def fingerprint_bits(pdbid, mol):
    import prolif as plf
    prot = protein(pdbid)
    lig = plf.Molecule.from_rdkit(mol)
    fp = plf.Fingerprint(count=False)
    fp.run_from_iterable([lig], prot, progress=False, n_jobs=1)
    df = fp.to_dataframe()
    bits = set()
    if len(df):
        row = df.iloc[0]
        for col in df.columns:
            if bool(row[col]):
                res = str(col[1])
                bits.add((BACK.get(pdbid, {}).get(res, res), str(col[2])))
    return bits


def tanimoto_bits(a, b):
    u = len(a | b)
    return float("nan") if u == 0 else len(a & b) / u
