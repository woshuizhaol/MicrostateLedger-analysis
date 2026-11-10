from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise SystemExit("input not found")

    suppl = Chem.SDMolSupplier(str(p), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise SystemExit("failed to read mol")

    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    print(smi)


if __name__ == "__main__":
    main()
