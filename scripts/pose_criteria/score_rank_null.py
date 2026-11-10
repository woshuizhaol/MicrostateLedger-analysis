import glob, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT = Path(os.environ.get("CRITERIA_OUT_ROOT", str(ROOT)))
RES = OUT / "results/pose_criteria"; RES.mkdir(parents=True, exist_ok=True)
pc = pd.read_csv(RES / "casf_pose_checks.csv", dtype={"pdbid": str, "microstate_id": str, "seed": str}, keep_default_na=False, na_values=[""])
ms = pd.read_csv(ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv", dtype=str, keep_default_na=False)
single, dom = {}, {}
for r in ms.itertuples():
    roles = r.roles.split(";")
    if "single" in roles: single[r.pdbid] = r.microstate_id
    if "dominant" in roles: dom[r.pdbid] = r.microstate_id
aff = sorted(p for p in single if single[p] != dom[p])
n_mic = pc.groupby("pdbid").microstate_id.nunique()
rows = []
for p in aff:
    n = int(n_mic[p]); k = math.ceil(n / 2)
    chance = 2.0 * k * (n - k) / (n * (n - 1))
    flips, better, pairs = 0, 0, 0
    for sd, g in pc[pc.pdbid == p].groupby("seed"):
        a = g[g.microstate_id == single[p]].iloc[0]; b = g[g.microstate_id == dom[p]].iloc[0]
        pairs += 1; flips += bool(a.score_rank) != bool(b.score_rank); better += float(b.vina_score) < float(a.vina_score)
    rows.append(dict(pdbid=p, n_microstates=n, chance_flip=chance, observed_flip=flips / pairs, dominant_better=better / pairs))
t = pd.DataFrame(rows)
bins = {"n=2": t.n_microstates == 2, "n=3-5": t.n_microstates.between(3, 5), "n=6-10": t.n_microstates.between(6, 10), "n>=11": t.n_microstates >= 11}
by_n = {k: dict(n_complexes=int(m.sum()), chance_flip_pct=100 * float(t[m].chance_flip.mean()), observed_flip_pct=100 * float(t[m].observed_flip.mean())) for k, m in bins.items() if m.any()}
res = dict(n_affected_complexes=len(t), microstates_per_complex=dict(median=float(t.n_microstates.median()), mean=float(t.n_microstates.mean()), min=int(t.n_microstates.min()), max=int(t.n_microstates.max()),
           n_complexes_with_2=int((t.n_microstates == 2).sum())), chance_flip_pct=100 * float(t.chance_flip.mean()), observed_flip_pct=100 * float(t.observed_flip.mean()),
           dominant_better_than_single_pct=100 * float(t.dominant_better.mean()), by_number_of_microstates=by_n,
           provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_criteria.yaml", [RES / "casf_pose_checks.csv", ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv"], seed=None))
json.dump(res, open(RES / "casf_score_rank_null.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
