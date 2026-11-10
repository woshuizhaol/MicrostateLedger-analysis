from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import RegistrationHash
from rdkit.Chem.inchi import MolToInchiKey


def _read_input(inp: str) -> Chem.Mol:
    p = Path(inp)
    if p.exists():
        if p.suffix.lower() in {".sdf", ".mol"}:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        else:
            text = p.read_text(encoding="utf-8").strip().splitlines()
            if not text:
                raise ValueError("Empty input file")
            smi = text[0].split()[0]
            mol = Chem.MolFromSmiles(smi)
    else:
        mol = Chem.MolFromSmiles(inp)
    if mol is None:
        raise ValueError("Failed to read molecule")
    return mol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--keep-salts", action="store_true", default=False)
    args = ap.parse_args()

    mol = _read_input(args.input)

    if args.standardize:
        mol = rdMolStandardize.Cleanup(mol)
        if not args.keep_salts:
            mol = rdMolStandardize.FragmentParent(mol)
            mol = rdMolStandardize.ChargeParent(mol)
            te = rdMolStandardize.TautomerEnumerator()
            mol = te.Canonicalize(mol)

    layers = RegistrationHash.GetMolLayers(mol)
    reg_hash = RegistrationHash.GetMolHash(layers)
    inchikey = MolToInchiKey(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_path))
    w.write(mol)
    w.close()

    print(json.dumps({
        "reg_hash": reg_hash,
        "inchikey": inchikey,
        "smiles": smiles,
        "formal_charge": Chem.GetFormalCharge(mol),
    }))


if __name__ == "__main__":
    main()
