import os, sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P
import pose_flip as F

OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RES, RAW = OUT / "results/pose_validity", OUT / "raw_outputs/pose_validity"
lt = pd.read_csv(RES / "full_scale_ligand_flips.csv", dtype={"ligand_id": str, "uid": str}, keep_default_na=True)
pose = pd.read_parquet(RAW / "full_scale_pose_table.parquet", columns=["uid", "microstate_id", "pose_id", "pb_nan_columns"])
pose = pose.drop_duplicates("pose_id")
pose["nan_energy"] = pose["pb_nan_columns"].astype(str).str.contains("internal_energy") & pose["pb_nan_columns"].notna()
nan_pose = set(zip(pose.loc[pose.nan_energy, "uid"], pose.loc[pose.nan_energy, "microstate_id"]))
nan_uid = set(pose.loc[pose.nan_energy, "uid"])
print("poses %d, with internal_energy NaN %d (%.3f%%), ligands (uid) touched %d" % (len(pose), pose.nan_energy.sum(), 100 * pose.nan_energy.mean(), len(nan_uid)), flush=True)

lt["default_nan"] = [(u, m) in nan_pose for u, m in zip(lt["uid"], lt["default_microstate"])]
lt["best_nan"] = [(u, m) in nan_pose for u, m in zip(lt["uid"], lt["best_microstate"])]
lt["any_nan"] = lt["uid"].isin(nan_uid)
lt["direct_nan"] = lt["default_nan"] | lt["best_nan"]
variants = {"all": lt, "excl_direct": lt[~lt["direct_nan"]], "excl_any": lt[~lt["any_nan"]]}

rows = []
for name, d in variants.items():
    for scope, dd in (("by_role", d), ("ALL", d.assign(role="ALL"))):
        fr = F.flip_rates(dd, ["set", "role"])
        fr["variant"] = name
        rows.append(fr)
tab = pd.concat(rows, ignore_index=True)
n_total = lt[lt.has_default.astype(bool) & (lt.n_micro >= 2)].groupby(["set", "role"]).size().rename("n_primary_all").reset_index()
tab = tab.merge(n_total, on=["set", "role"], how="left")
allrole = lt[lt.has_default.astype(bool) & (lt.n_micro >= 2)].assign(role="ALL").groupby(["set", "role"]).size().rename("n_primary_all_b").reset_index()
tab = tab.merge(allrole, on=["set", "role"], how="left")
tab["n_primary_all"] = tab["n_primary_all"].fillna(tab["n_primary_all_b"]).astype(int)
tab = tab.drop(columns=["n_primary_all_b"])
cols = ["variant", "set", "role", "criterion", "denominator", "n_ligands", "n_primary_all", "n_flip", "flip_pct", "wilson_lo", "wilson_hi", "n_rescue", "n_degrade"]
tab = tab[cols].sort_values(["set", "role", "criterion", "denominator", "variant"]).reset_index(drop=True)
o1 = RES / "posthoc_exclusion_nan_energy.csv"
tab.to_csv(o1, index=False, float_format="%.6g")

prim = lt[lt.has_default.astype(bool) & (lt.n_micro >= 2)]
cnt = []
for (s, r), g in pd.concat([prim, prim.assign(role="ALL")]).groupby(["set", "role"]):
    cnt.append(dict(set=s, role=r, n_ligands_primary=len(g), n_default_or_best_nan=int(g.direct_nan.sum()), n_any_microstate_nan=int(g.any_nan.sum()),
                    pct_default_or_best_nan=100 * g.direct_nan.mean(), pct_any_microstate_nan=100 * g.any_nan.mean()))
cn = pd.DataFrame(cnt)
o2 = RES / "posthoc_nan_energy_ligands.csv"
cn.to_csv(o2, index=False, float_format="%.6g")

BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [RES / "full_scale_ligand_flips.csv", RAW / "full_scale_pose_table.parquet"], seed=None, extra=dict(post_hoc=True, n_poses=int(len(pose)), n_nan_energy_poses=int(pose.nan_energy.sum())))
P.write_schema(o1, [("variant", "", "all / excl_direct / excl_any (see the script header)"), ("set", "", "ligand set"), ("role", "", "active / negative / ALL"), ("criterion", "", "geom | full_pb | score_rank"),
                    ("denominator", "", "primary = ligands with a default and >= 2 docked microstates; best_differs = those whose best microstate is not the default"), ("n_ligands", "", "ligands of the denominator in this variant"),
                    ("n_primary_all", "", "ligands of the primary denominator before any exclusion (same set and role)"), ("n_flip", "", "ligands whose default and best microstate poses differ in validity"), ("flip_pct", "%", "100 * n_flip / n_ligands"),
                    ("wilson_lo", "%", "Wilson 95% lower"), ("wilson_hi", "%", "Wilson 95% upper"), ("n_rescue", "", "default invalid -> best valid"), ("n_degrade", "", "default valid -> best invalid")],
               "one row per variant x set x role x criterion x denominator", "POST HOC: validity flip rates of the decoy-matching poses without the ligands whose PoseBusters energy-ratio check gave no number", extra=dict(provenance=BLOCK))
P.write_schema(o2, [("set", "", "ligand set"), ("role", "", "active / negative / ALL"), ("n_ligands_primary", "", "ligands with a default and >= 2 docked microstates"), ("n_default_or_best_nan", "", "of them: default or best microstate pose has internal_energy NaN"),
                    ("n_any_microstate_nan", "", "of them: any docked microstate pose has internal_energy NaN"), ("pct_default_or_best_nan", "%", "share of n_ligands_primary"), ("pct_any_microstate_nan", "%", "share of n_ligands_primary")],
               "one row per set x role", "POST HOC: ligands touched by poses with internal_energy NaN", extra=dict(provenance=BLOCK))
print(cn.round(2).to_string(index=False))
sel = tab[(tab.role == "ALL") & (tab.denominator == "primary") & (tab.criterion.isin(["geom", "full_pb"]))]
print(sel.pivot_table(index=["set", "criterion"], columns="variant", values="flip_pct").round(2).to_string())
print("wrote %s (%d rows) and %s (%d rows)" % (o1, len(tab), o2, len(cn)))
