from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--prune_rmsd", type=float, default=0.5)
    ap.add_argument("--select-k", type=int, default=0, help="Select top-k conformers")
    ap.add_argument("--cluster-threshold", type=float, default=0.0, help="RMSD threshold for diversity clustering")
    ap.add_argument("--seed", type=int, default=-1, help="Random seed for ETKDG (-1 = random)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suppl = Chem.SDMolSupplier(str(args.input), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise SystemExit("failed to read mol")

    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = args.prune_rmsd
    if args.seed is not None and int(args.seed) >= 0:
        params.randomSeed = int(args.seed)
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=args.n, params=params)

    energies = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            ff.Minimize()
            energies.append(ff.CalcEnergy())
    else:
        for cid in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            ff.Minimize()
            energies.append(ff.CalcEnergy())

    out_records = []
    for i, cid in enumerate(conf_ids):
        mcopy = Chem.Mol(mol)
        mcopy.RemoveAllConformers()
        mcopy.AddConformer(mol.GetConformer(cid), assignId=True)
        sdf_path = out_dir / f"conf_{i}.sdf"
        if sdf_path.exists() and sdf_path.is_symlink():
            sdf_path.unlink()
        w = Chem.SDWriter(str(sdf_path))
        w.write(mcopy)
        w.close()
        out_records.append({"sdf_path": str(sdf_path), "energy": float(energies[i]) if i < len(energies) else 0.0})

    selected = out_records
    cluster_used = False
    if args.select_k and args.select_k > 0 and len(out_records) > args.select_k:
        if args.cluster_threshold and args.cluster_threshold > 0 and len(conf_ids) > 1:
            heavy = Chem.RemoveHs(mol)
            dists = []
            for i in range(len(conf_ids)):
                for j in range(i):
                    dists.append(rdMolAlign.CalcRMS(heavy, heavy, conf_ids[i], conf_ids[j]))
            clusters = Butina.ClusterData(dists, len(conf_ids), args.cluster_threshold, isDistData=True)
            cluster_reps = []
            for cl in clusters:
                best = min(cl, key=lambda idx: energies[idx] if idx < len(energies) else 0.0)
                cluster_reps.append(best)
            cluster_reps_sorted = sorted(cluster_reps, key=lambda idx: energies[idx] if idx < len(energies) else 0.0)
            keep = cluster_reps_sorted[: args.select_k]
            selected = [out_records[i] for i in keep]
            cluster_used = True
        else:
            selected = sorted(out_records, key=lambda c: float(c.get("energy", 0.0)))[: args.select_k]

    payload = {
        "conformers": selected,
        "n_total": len(out_records),
        "n_selected": len(selected),
        "cluster_used": cluster_used,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
