import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RES = OUT / "results/pose_validity"
N = {"dude_full": (20, 40), "litpcba": (10, 20), "zinc_unmatched": (0, 30)}
rng = np.random.default_rng(20261032)
lt = pd.read_csv(RES / "full_scale_ligand_flips.csv", dtype={"ligand_id": str})
maxs = int(sys.argv[sys.argv.index("--max-slice") + 1]) if "--max-slice" in sys.argv else None
if maxs is not None:
    q = pd.read_csv(ROOT / "raw_outputs/decoy_matching/queue/ligand_sets_sliced.csv", dtype=str, keep_default_na=False)
    import hashlib
    q["uid"] = [hashlib.md5(("%s|%s" % (t, s_)).encode()).hexdigest()[:12] for t, s_ in zip(q.target, q.smiles)]
    sl = q.drop_duplicates(["set", "uid"]).set_index(["set", "uid"])["slice"].astype(int)
    lt["slice"] = [sl.get((s_, u), 99) for s_, u in zip(lt["set"], lt["uid"])]
    lt = lt[lt.slice <= maxs]
elig = lt[lt.has_default.astype(bool) & (lt.n_micro >= 2)].copy()
elig = elig[~elig.best_is_default.astype(bool)]
rows = []
for (s_name, t), g in elig.groupby(["set", "target"]):
    if s_name not in N:
        continue
    na, nn = N[s_name]
    for role, k in (("active", na), ("negative", nn)):
        gg = g[g.role == role]
        if k and len(gg):
            take = gg.iloc[rng.choice(len(gg), size=min(k, len(gg)), replace=False)]
            rows.append(take)
smp = pd.concat(rows, ignore_index=True)
rec = {}
import glob
for f in glob.glob(str(ROOT / "raw_outputs/decoy_matching/dock/*.jsonl")):
    if f.endswith("errors.jsonl"):
        continue
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        for m in r["microstates"]:
            if m["status"] == "ok":
                rec[(r["uid"], m["microstate_id"])] = (m["conf_rank"], m["seed"], m["conf_energy"])
for lab in ("default", "best"):
    for k, nm in enumerate(("conf_rank", "seed", "conf_energy")):
        smp["%s_%s" % (lab, nm)] = [rec.get((u, m), (None, None, None))[k] for u, m in zip(smp["uid"], smp[lab + "_microstate"])]
q2 = pd.read_csv(ROOT / "raw_outputs/decoy_matching/queue/ligand_sets.csv", dtype=str, keep_default_na=False)
import hashlib
q2["uid"] = [hashlib.md5(("%s|%s" % (t, sm)).encode()).hexdigest()[:12] for t, sm in zip(q2["target"], q2["smiles"])]
smp = smp.merge(q2.drop_duplicates("uid")[["uid", "smiles"]], on="uid", how="left")
cols = ["set", "target", "role", "round", "ligand_id", "uid", "smiles", "n_micro", "default_microstate", "best_microstate", "default_conf_rank", "best_conf_rank", "default_seed", "best_seed", "default_conf_energy", "best_conf_energy", "default_score", "best_score", "default_geom", "default_full_pb", "default_score_rank",
        "best_geom", "best_full_pb", "best_score_rank", "flip_geom", "flip_full_pb", "flip_score_rank"]
smp = smp[cols]
RES.mkdir(parents=True, exist_ok=True)
path = RES / "noise_sample.csv"
smp.to_csv(path, index=False)
desc = {c: "see noise_sample.py (columns of full_scale_ligand_flips.csv plus conformer ranks and seeds of the decoy-matching records)" for c in cols}
P.write_schema(path, [(c, "", desc[c]) for c in cols], "one row per sampled ligand", "Ligands re-docked with extra Vina seeds for the noise floor", extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [RES / "full_scale_ligand_flips.csv"], seed=20261032, extra=dict(max_slice=maxs))))
print(len(smp), "ligands sampled;", smp.groupby(["set", "role"]).size().to_dict(), "| missing conformer rank:", int(smp["default_conf_rank"].isna().sum() + smp["best_conf_rank"].isna().sum()))
