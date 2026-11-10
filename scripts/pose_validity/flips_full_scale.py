import glob, hashlib, json, os, sys, time
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
DECOY_DIR = Path(os.environ.get("DECOY_ROOT", str(ROOT)))
RAW, RES = OUT / "raw_outputs/pose_validity", OUT / "results/pose_validity"; RES.mkdir(parents=True, exist_ok=True)
MAX_ROUND = int(os.environ.get("POSE_MAX_ROUND", "3"))
SEED, N_BOOT = 20261031, 2000


def uid_of(target, smiles):
    return hashlib.md5(("%s|%s" % (target, smiles)).encode()).hexdigest()[:12]


t0 = time.time()
q = pd.read_csv(DECOY_DIR / "raw_outputs/decoy_matching/queue/ligand_sets.csv", dtype=str, keep_default_na=False)
q["round"] = q["round"].astype(int)
q["uid"] = [uid_of(t, s) for t, s in zip(q.target, q.smiles)]
act = q[(q.set == "dude_full") & (q.role == "active")]
extra = pd.concat([act.assign(set=s) for s in ("zinc_unmatched", "zinc_charge_mismatch")], ignore_index=True)
q = pd.concat([q, extra], ignore_index=True)
q = q[q["round"] <= MAX_ROUND]
parts = sorted(glob.glob(str(RAW / "bust/new/*.parquet")))
bust = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True).drop_duplicates("pose_id") if parts else pd.DataFrame(columns=["pose_id"])
rec = []
for f in sorted(glob.glob(str(DECOY_DIR / "raw_outputs/decoy_matching/dock/*.jsonl"))):
    if f.endswith("errors.jsonl"):
        continue
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        for m in r["microstates"]:
            rec.append(dict(uid=r["uid"], target=r["target"], order=m["order"], microstate_id=m["microstate_id"], formal_charge=m["formal_charge"], stereo_tag=m["smiles"], tautomer_key=m["tautomer_key"],
                            status=m["status"], stage=m.get("stage", ""), vina_score=m.get("score"), conf_rank=m.get("conf_rank"), n_eligible=r["n_eligible"]))
pose = pd.DataFrame(rec)
pose["pose_id"] = pose.uid + "|" + pose.microstate_id
n_ms = len(pose)
ok = pose[pose.status == "ok"].merge(bust[["pose_id", "geom", "full_pb", "inter", "failed_checks", "pb_nan_columns"]], on="pose_id", how="left")
has_v = ok.groupby("uid").full_pb.apply(lambda s: s.notna().all())
n_dock_fail = pose.groupby("uid").status.apply(lambda s: int((s != "ok").sum()))
complete = has_v[has_v].index
ok = ok[ok.uid.isin(complete)].copy()
for c in ("geom", "full_pb", "inter"):
    ok[c] = ok[c].astype(bool)
ok["vina_score"] = ok["vina_score"].astype(float)
ok["is_default"] = ok["order"] == 0
LIG = ["set", "target", "role", "round", "ligand_id", "uid"]
ok = ok.drop(columns=["target"]).merge(q[LIG], on="uid", how="inner")
lt = F.ligand_table(ok.assign(conf_rank=0), LIG)
lt = lt.merge(pd.DataFrame({"uid": n_dock_fail.index, "n_microstates_failed": n_dock_fail.values}), on="uid", how="left")
lt.to_csv(RES / "full_scale_ligand_flips.csv", index=False)
fr = pd.concat([F.flip_rates(lt, ["set", "target", "role"]), F.flip_rates(lt.assign(target="ALL"), ["set", "target", "role"]), F.flip_rates(lt.assign(target="ALL", role="ALL"), ["set", "target", "role"])], ignore_index=True)
fr.to_csv(RES / "full_scale_flip_rate_by_criterion.csv", index=False)
rng = np.random.default_rng(SEED)
pl = F.add_pose_level_rank(ok, LIG)
rows = []
for s, sub in pl.groupby("set"):
    sub = sub.assign(lig=sub.target + "|" + sub.uid)
    for x, y in CFG["kappa"]["pairs"]:
        rows.append(dict(set=s, unit="all_poses", criterion_a=x, criterion_b=y, **F.kappa_cluster(sub, x, y, ["lig"], N_BOOT, rng)))
pd.DataFrame(rows).to_csv(RES / "full_scale_criteria_kappa.csv", index=False)
pl.to_parquet(RAW / "full_scale_pose_table.parquet")
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [DECOY_DIR / "raw_outputs/decoy_matching/queue/ligand_sets.csv"], seed=SEED, n_boot=N_BOOT,
                           extra=dict(n_bust_parquet_files=len(parts), max_round=MAX_ROUND))
F.write_schemas(P, [(RES / "full_scale_ligand_flips.csv", "one row per (set, target, ligand)", "Default vs best-scoring microstate validity of the decoy-matching poses"),
                    (RES / "full_scale_flip_rate_by_criterion.csv", "one row per (set, target, role) x criterion x denominator", "Validity flip rates of the decoy-matching poses (Wilson CI)"),
                    (RES / "full_scale_criteria_kappa.csv", "one row per set x criterion pair", "Pairwise Cohen's kappa of geom / full_pb / score_rank on the decoy-matching poses")], BLOCK)
summ = dict(provenance=BLOCK, n_microstate_dockings_recorded=n_ms, n_dock_failed=int((pose.status != "ok").sum()), n_poses_with_verdict=int(len(ok)), n_ligand_set_rows=len(lt), n_records_waiting_for_bust=int((~has_v).sum()),
            pass_rate={s: {c: float(g[c].mean()) for c in ("geom", "full_pb", "inter")} for s, g in ok.groupby("set")}, failed_checks_top=pd.Series([c for s in ok.loc[~ok.full_pb, "failed_checks"] for c in s.split(";") if c]).value_counts().head(10).to_dict(),
            seconds=round(time.time() - t0, 1))
json.dump(summ, open(RES / "full_scale_pose_checks_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
if len(fr):
    print(fr[(fr.target == "ALL") & (fr.role == "ALL") & (fr.denominator == "primary")][["set", "criterion", "n_ligands", "n_flip", "flip_pct", "wilson_lo", "wilson_hi", "n_rescue", "n_degrade"]].round(1).to_string(index=False))
