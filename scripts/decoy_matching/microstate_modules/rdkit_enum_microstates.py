from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Lipinski, QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def microstate_id(fixedh_inchi: str, formal_charge: int, stereo_tag: str, coordination_sig: str = "") -> str:
    payload = f"{fixedh_inchi}|{formal_charge}|{stereo_tag}|{coordination_sig}"
    return "M-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


METAL_ATOMIC_NUMS = set([
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
])

COORDINATION_MOTIFS = [
    ("carboxylate", Chem.MolFromSmarts("[CX3](=O)[O-]")),
    ("carboxylic_acid", Chem.MolFromSmarts("[CX3](=O)[OH]")),
    ("hydroxamate", Chem.MolFromSmarts("[CX3](=O)N[O-]")),
    ("catechol", Chem.MolFromSmarts("c1ccc(O)c(O)cc1")),
    ("beta_diketonate", Chem.MolFromSmarts("O=C-C(=O)")),
    ("pyridine", Chem.MolFromSmarts("n1ccccc1")),
    ("imidazole", Chem.MolFromSmarts("n1cncc1")),
    ("bipyridine", Chem.MolFromSmarts("n1ccccn1")),
    ("phenanthroline", Chem.MolFromSmarts("n1ccc2ncccc2c1")),
    ("thiol", Chem.MolFromSmarts("[SX2H]")),
    ("thioether", Chem.MolFromSmarts("[SX2]([#6])[#6]")),
    ("phosphine", Chem.MolFromSmarts("[PX3]")),
]


def coordination_signature(mol: Chem.Mol) -> str:
    metals = []
    donors = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in METAL_ATOMIC_NUMS:
            metals.append(atom.GetSymbol())
        if atom.GetSymbol() in ("N", "O", "S", "P") and atom.GetFormalCharge() <= 0:
            donors.append(atom.GetSymbol())

    motifs = []
    for name, smarts in COORDINATION_MOTIFS:
        if smarts is None:
            continue
        if mol.HasSubstructMatch(smarts):
            motifs.append(name)

    if not metals and not donors and not motifs:
        return ""
    metals_str = ",".join(sorted(set(metals))) if metals else "none"
    donors_str = ",".join(sorted(set(donors))) if donors else "none"
    donor_count = len(donors)
    motifs_str = ",".join(sorted(set(motifs))) if motifs else "none"
    return f"metals:{metals_str};donors:{donors_str};donor_count:{donor_count};motifs:{motifs_str}"


def apply_atom_map(mol: Chem.Mol) -> list[dict]:
    ranks = list(Chem.CanonicalRankAtoms(mol))
    order = sorted(range(len(ranks)), key=lambda i: (ranks[i], i))
    mapping = []
    for current_id, atom_idx in enumerate(order, start=1):
        mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(current_id)
        mapping.append({
            "canonical_atom_id": current_id,
            "atom_index": atom_idx,
            "symbol": mol.GetAtomWithIdx(atom_idx).GetSymbol(),
        })
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-smiles-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tautomers", type=int, default=32)
    ap.add_argument("--max-isomers", type=int, default=16)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    smiles_list = []
    for line in Path(args.input_smiles_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        smiles_list.append(line.split()[0])

    te = rdMolStandardize.TautomerEnumerator()
    te.SetMaxTautomers(args.max_tautomers)
    uncharger = rdMolStandardize.Uncharger()

    opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, onlyUnassigned=True)

    seen = {}
    out_records = []
    stats = {
        "n_input_smiles": len(smiles_list),
        "tautomer_truncated": 0,
        "stereo_truncated": 0,
        "sanitize_failed": 0,
        "invalid_smiles": 0,
    }

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_smiles"] += 1
            continue
        base_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        tautomers = te.Enumerate(mol)
        if len(tautomers) > args.max_tautomers:
            stats["tautomer_truncated"] += int(len(tautomers) - args.max_tautomers)
        for taut in list(tautomers)[: args.max_tautomers]:
            canonical_taut = te.Canonicalize(taut)
            tautomer_key = Chem.MolToSmiles(canonical_taut, isomericSmiles=True)
            isomers = list(EnumerateStereoisomers(taut, options=opts))
            if not isomers:
                isomers = [taut]
            if len(isomers) > args.max_isomers:
                stats["stereo_truncated"] += int(len(isomers) - args.max_isomers)
            isomers = isomers[: args.max_isomers]
            for iso in isomers:
                try:
                    Chem.SanitizeMol(iso)
                except Exception:
                    stats["sanitize_failed"] += 1
                    continue
                try:
                    fixedh = Chem.MolToInchi(Chem.AddHs(iso, addCoords=False), options="/FixedH")
                    stereo_tag = Chem.MolToSmiles(iso, isomericSmiles=True)
                except Exception:
                    continue
                charge = Chem.GetFormalCharge(iso)
                try:
                    parent = rdMolStandardize.ChargeParent(iso)
                    parent = uncharger.uncharge(parent)
                    protonation_key = Chem.MolToSmiles(parent, isomericSmiles=True)
                except Exception:
                    protonation_key = stereo_tag
                try:
                    hbd = int(Lipinski.NumHDonors(iso))
                    hba = int(Lipinski.NumHAcceptors(iso))
                except Exception:
                    hbd, hba = 0, 0
                try:
                    rdkit_score = float(QED.qed(iso))
                except Exception:
                    rdkit_score = 0.0
                coord_sig = coordination_signature(iso)
                mid = microstate_id(fixedh, charge, stereo_tag, coord_sig)
                if mid in seen:
                    continue
                seen[mid] = True

                mdir = out_root / mid
                mdir.mkdir(parents=True, exist_ok=True)
                mol_copy = Chem.Mol(iso)
                mapping = apply_atom_map(mol_copy)
                sdf_path = mdir / "microstate.sdf"
                if not sdf_path.exists():
                    w = Chem.SDWriter(str(sdf_path))
                    w.write(mol_copy)
                    w.close()

                map_path = mdir / "atom_map.json"
                if not map_path.exists():
                    with map_path.open("w", encoding="utf-8") as f:
                        json.dump(mapping, f, ensure_ascii=False, indent=2)

                out_records.append({
                    "microstate_id": mid,
                    "fixedh_inchi": fixedh,
                    "formal_charge": charge,
                    "stereo_tag": stereo_tag,
                    "coordination_sig": coord_sig,
                    "sdf_path": str(sdf_path),
                    "atom_map_path": str(map_path),
                    "protonation_key": protonation_key,
                    "tautomer_key": tautomer_key,
                    "hbd": hbd,
                    "hba": hba,
                    "rdkit_score": rdkit_score,
                })

    print(json.dumps({"microstates": out_records, "stats": stats}))


if __name__ == "__main__":
    main()
