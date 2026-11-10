import json, os, sys, itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_flip as F

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RAW, RES = OUT / "raw_outputs/pose_validity", OUT / "results/pose_validity"
u = pd.read_parquet(RAW / "ten_target_rep_pose_table.parquet")
for c in ("geom", "full_pb"):
    u[c] = u[c].astype(bool)
key = ["dataset", "target", "compound_id", "microstate_id"]
g = u.groupby(key)
multi = g.seed.nunique()
multi = multi[multi >= 2]
print("units (compound x microstate) in >= 2 seeds:", len(multi), "of", g.ngroups, flush=True)
u = u.set_index(key).loc[multi.index].reset_index()
rows = []
for k, gg in u.groupby(key):
    vals = gg.sort_values("seed")
    for (_, a), (_, b) in itertools.combinations(vals.iterrows(), 2):
        rows.append(dict(zip(key, k), seed_a=a.seed, seed_b=b.seed, geom_differs=a.geom != b.geom, full_pb_differs=a.full_pb != b.full_pb, score_a=a.vina_score, score_b=b.vina_score))
pr = pd.DataFrame(rows)
rng = np.random.default_rng(20261034)
out = []
for ds, d in list(pr.groupby("dataset")) + [("ALL", pr)]:
    comp = d.groupby(["dataset", "target", "compound_id"])
    for c in ("geom", "full_pb"):
        k, n = int(d[c + "_differs"].sum()), len(d)
        lo, hi = F.wilson(k, n)
        s = comp[c + "_differs"].agg(["sum", "count"]).values
        idx = rng.integers(0, len(s), size=(2000, len(s)))
        tot = s[idx].sum(1)
        r = tot[:, 0] / tot[:, 1]
        out.append(dict(dataset=ds, criterion=c, n_seed_pairs=n, n_units=d.groupby(key).ngroups, n_compounds=len(s), n_disagree=k, disagree_rate=k / n, wilson_lo=lo, wilson_hi=hi, boot_lo=float(np.percentile(r, 2.5)), boot_hi=float(np.percentile(r, 97.5)),
                        mean_abs_score_diff=float((d.score_a - d.score_b).abs().mean())))
res = pd.DataFrame(out)
path = RES / "ten_target_seed_overlap_noise.csv"; res.to_csv(path, index=False, float_format="%.6g")
desc = {"dataset": "ten_target_main | ten_target_budget6 | ALL", "criterion": "geom | full_pb", "n_seed_pairs": "pairs of seeds in which the same compound/microstate was docked", "n_units": "compound x microstate units", "n_compounds": "compounds (bootstrap clusters)",
        "n_disagree": "pairs whose representative-pose verdict differs", "disagree_rate": "n_disagree / n_seed_pairs", "wilson_lo": "Wilson 95% lower (pairs treated as independent; descriptive)", "wilson_hi": "Wilson 95% upper",
        "boot_lo": "compound-cluster bootstrap 2.5%", "boot_hi": "compound-cluster bootstrap 97.5%", "mean_abs_score_diff": "mean |Vina score difference| between the two seeds (kcal/mol)"}
P.write_schema(path, [(c, "kcal/mol" if c == "mean_abs_score_diff" else "", desc[c]) for c in res.columns], "one row per dataset x criterion", "Seed-to-seed disagreement of the validity of the same microstate in the original data",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [RAW / "ten_target_rep_pose_table.parquet"], seed=20261034, n_boot=2000)))
print(res.round(4).to_string(index=False))
