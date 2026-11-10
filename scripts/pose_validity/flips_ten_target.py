import csv, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_flip as F

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/pose_validity.yaml", encoding="utf-8"))
OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RAW, RES = OUT / "raw_outputs/pose_validity", OUT / "results/pose_validity"; RES.mkdir(parents=True, exist_ok=True)
PARTIAL = "--allow-partial" in sys.argv
NB = CFG["kappa"]["ci"]
N_BOOT, SEED = 2000, 20261031
LIG = ["dataset", "seed", "target", "ligand_idx"]

man = pd.read_csv(RAW / "ten_target_pose_manifest.csv", dtype={"seed": str})
def read_part(f):
    ds, sd = Path(f).name.split("__")[:2]
    return pd.read_parquet(f).assign(dataset=ds, seed=sd)


KEY = ["dataset", "seed", "pose_id"]
parts = [read_part(f) for f in sorted(glob.glob(str(RAW / "bust/original/*.parquet")))]
bust = pd.concat(parts, ignore_index=True)
n_dup_verdict_rows = int(bust.duplicated(KEY).sum())
bust = bust.drop_duplicates(KEY)
assert not man.duplicated(KEY).any(), "manifest key (dataset, seed, pose_id) is not unique"
d = man.merge(bust[KEY + ["geom", "full_pb", "inter", "failed_checks", "pb_nan_columns"]], on=KEY, how="left", validate="one_to_one")
n_missing = int(d.full_pb.isna().sum())
cover = d.groupby(LIG).full_pb.apply(lambda s: s.notna().all())
if n_missing and not PARTIAL:
    sys.exit("%d of %d poses have no PoseBusters verdict yet (use --allow-partial for a partial test)" % (n_missing, len(d)))
d = d.merge(cover.rename("ligand_complete").reset_index(), on=LIG)
d = d[d.ligand_complete].copy()
for c in ("geom", "full_pb", "inter"):
    d[c] = d[c].astype(bool)
d["vina_score"] = d["vina_score"].astype(float)

defaults = {}
for (ds, sd, t), g in d.groupby(["dataset", "seed", "target"]):
    tdir = Path(g.sdf_path.iloc[0]).parent.parent
    for r in csv.DictReader(open(tdir / "rows.csv", encoding="utf-8")):
        try:
            al = json.loads(r.get("strategy_allocation_json") or "{}")
        except json.JSONDecodeError:
            continue
        s = al.get("single")
        if isinstance(s, dict) and s.get("candidate_mids"):
            defaults[(ds, sd, t, int(r["idx"]))] = s["candidate_mids"][0]
d["default_mid"] = [defaults.get((a, b, c, e)) for a, b, c, e in zip(d.dataset, d.seed, d.target, d.ligand_idx)]
d["is_default"] = d.microstate_id == d.default_mid

rep = F.representative(d, LIG)
lt = F.ligand_table(rep, LIG)
meta = d.drop_duplicates(LIG)[LIG + ["compound_id", "label", "source"]]
lt = lt.merge(meta, on=LIG, how="left")
lt.to_csv(RES / "ten_target_ligand_flips.csv", index=False)
fr_t = F.flip_rates(lt, ["dataset", "seed", "target"])
lt_all = lt.assign(target="ALL")
fr_all = F.flip_rates(lt_all, ["dataset", "seed", "target"])
fr_pool = F.flip_rates(lt.assign(seed="ALL", target="ALL"), ["dataset", "seed", "target"])
fr_ds_t = F.flip_rates(lt.assign(seed="ALL"), ["dataset", "seed", "target"])
fr = pd.concat([fr_t, fr_all, fr_ds_t, fr_pool], ignore_index=True).drop_duplicates()
fr.to_csv(RES / "ten_target_flip_rate_by_criterion.csv", index=False)

rng = np.random.default_rng(SEED)
dp = F.add_pose_level_rank(d, LIG)
rep_p = rep.copy(); rep_p["score_rank"] = False
for _, ix in rep_p.groupby(LIG).indices.items():
    rep_p.iloc[ix, rep_p.columns.get_loc("score_rank")] = F.score_rank_pass(rep_p.iloc[ix]["vina_score"].values)
rows = []
for ds in sorted(d.dataset.unique()):
    for unit, frame in (("all_poses", dp), ("representative_poses", rep_p)):
        sub = frame[frame.dataset == ds]
        sub = sub.assign(lig=sub.seed.astype(str) + "|" + sub.target + "|" + sub.ligand_idx.astype(str))
        for x, y in CFG["kappa"]["pairs"]:
            r = F.kappa_cluster(sub, x, y, ["lig"], N_BOOT, rng)
            rows.append(dict(dataset=ds, unit=unit, criterion_a=x, criterion_b=y, **r))
pd.DataFrame(rows).to_csv(RES / "ten_target_criteria_kappa.csv", index=False)
rep_p.to_parquet(RAW / "ten_target_rep_pose_table.parquet")

rows2 = []
base = Path(CFG["paths"]["ten_target_run"].replace("$DOCKING_ROOT", os.environ.get("DOCKING_ROOT", "."))) / "seed_20260210/docking_run"
_d018 = d[(d.dataset == "ten_target_main") & (d.seed == "20260210")]
valid = dict(zip(_d018.pose_id, _d018.full_pb))
for t in sorted(d[(d.dataset == "ten_target_main") & (d.seed == "20260210")].target.unique()):
    n = nf = nfa = 0
    for r in csv.DictReader(open(base / t / "rows.csv", encoding="utf-8")):
        st = {}
        for s_ in ("single", "top2", "full", "prioritized", "prioritized_adaptive"):
            pth = r.get("pose_sdf_" + s_)
            if not isinstance(pth, str) or not pth.strip():
                continue
            m = os.path.basename(pth)[:-4].split("_")
            pid = "ten_target_main|%s|%d|%s|%d" % (t, int(m[0]), m[1], int(m[2]))
            if pid in valid:
                st[s_] = (m[1], valid[pid])
        if not st:
            continue
        n += 1
        mids = {v[0]: v[1] for v in st.values()}
        if len(mids) >= 2 and any(mids.values()) and not all(mids.values()):
            nf += 1
    rows2.append(dict(target=t, n_ligands_checked=n, n_flip_old_definition=nf))
p2 = pd.DataFrame(rows2)
if len(p2):
    p2["flip_pct_old_definition"] = 100 * p2.n_flip_old_definition / p2.n_ligands_checked
    new = fr[(fr.dataset == "ten_target_main") & (fr.seed == "20260210") & (fr.criterion == "full_pb") & (fr.denominator == "primary") & (fr.target != "ALL")][["target", "n_ligands", "n_flip", "flip_pct"]].rename(columns={"n_ligands": "n_ligands_new_def", "n_flip": "n_flip_new_def", "flip_pct": "flip_pct_new_def"})
    p2 = p2.merge(new, on="target", how="left")
p2.to_csv(RES / "ten_target_definition_flip.csv", index=False)

BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [RAW / "ten_target_pose_manifest.csv"], seed=SEED, n_boot=N_BOOT,
                           extra=dict(n_bust_parquet_files=len(parts), n_poses_missing_verdict=n_missing, partial=PARTIAL, n_duplicate_verdict_rows_same_key=n_dup_verdict_rows,
                                  n_pose_ids_shared_by_two_seeds=int(man.pose_id.duplicated().sum())))
F.write_schemas(P, [(RES / "ten_target_ligand_flips.csv", "one row per (dataset, seed, target, ligand)", "Default vs best-scoring microstate validity of the original poses"),
                    (RES / "ten_target_flip_rate_by_criterion.csv", "one row per (dataset, seed, target) x criterion x denominator", "Validity flip rates of the original poses (Wilson CI)"),
                    (RES / "ten_target_criteria_kappa.csv", "one row per dataset x unit x criterion pair", "Pairwise Cohen's kappa of geom / full_pb / score_rank on the original poses"),
                    (RES / "ten_target_definition_flip.csv", "one row per target of ten-target docking seed 20260210", "Old p2 flip definition recomputed with full_pb next to the new definition")], BLOCK)
pct = {c: float(d.groupby("dataset")[c].mean().mean()) for c in ("geom", "full_pb", "inter")}
summ = dict(n_poses_manifest=len(man), n_poses_missing_verdict=n_missing, partial=PARTIAL, n_poses_used=len(d), n_ligands=len(lt), pass_rate_by_dataset={k: {c: float(v[c].mean()) for c in ("geom", "full_pb", "inter")} for k, v in d.groupby("dataset")},
            n_ligands_with_default=int(lt.has_default.sum()), n_ligands_missing_default=int((~lt.has_default).sum()), failed_checks_top=pd.Series([c for s in d.loc[~d.full_pb, "failed_checks"] for c in s.split(";") if c]).value_counts().head(10).to_dict())
summ["provenance"] = BLOCK
json.dump(summ, open(RES / "ten_target_pose_checks_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
print(fr[(fr.target == "ALL") & (fr.seed == "ALL") & (fr.denominator == "primary")][["dataset", "criterion", "n_ligands", "n_flip", "flip_pct", "wilson_lo", "wilson_hi"]].round(1).to_string(index=False))
