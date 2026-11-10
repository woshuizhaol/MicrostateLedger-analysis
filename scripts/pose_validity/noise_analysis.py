import glob, itertools, json, math, os, sys
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

OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RES, RAW = OUT / "results/pose_validity", OUT / "raw_outputs/pose_validity"
CFG = yaml.safe_load(open(ROOT / "configs/pose_validity.yaml", encoding="utf-8"))["noise_floor"]
SEEDS = [int(CFG["seed_base"]) + k for k in range(1, int(CFG["n_extra_seeds"]) + 1)]
CRIT = F.CRITERIA
N_BOOT, BSEED = 2000, 20261033

smp = pd.read_csv(RES / "noise_sample.csv", dtype={"ligand_id": str}).drop_duplicates(["set", "uid"])
uids = set(smp.uid)
noise = {}
for f in glob.glob(str(RAW / "noise/*.jsonl")):
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        noise[r["uid"]] = r
parts = sorted(glob.glob(str(RAW / "bust/noise/*.parquet")))
bust = pd.concat([pd.read_parquet(f, columns=["pose_id", "geom", "full_pb"]) for f in parts], ignore_index=True).drop_duplicates("pose_id") if parts else pd.DataFrame(columns=["pose_id", "geom", "full_pb"])
bv = {r.pose_id: (r.geom, r.full_pb) for r in bust.itertuples()}
scores = {}
for f in glob.glob(str(ROOT / "raw_outputs/decoy_matching/dock/*.jsonl")):
    if f.endswith("errors.jsonl"):
        continue
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        if r["uid"] in uids:
            scores[r["uid"]] = {m["microstate_id"]: float(m["score"]) for m in r["microstates"] if m["status"] == "ok"}


def rank_pass(sc, mid):
    n = len(sc)
    return (1 + sum(1 for v in sc.values() if v < sc[mid])) <= math.ceil(n / 2)


rows, excluded = [], dict(no_noise_record=0, noise_not_ok=0, units_incomplete=0, missing_verdict=0, score_mismatch=0, n_micro_mismatch=0)
for r in smp.itertuples():
    nz = noise.get(r.uid)
    if nz is None:
        excluded["no_noise_record"] += 1; continue
    if nz["status"] != "ok":
        excluded["noise_not_ok"] += 1; continue
    units = {u["label"]: u for u in nz["units"]}
    if set(units) != {"default", "best"} or any(u.get("status") != "ok" for u in units.values()):
        excluded["units_incomplete"] += 1; continue
    sc0 = scores.get(r.uid, {})
    if len(sc0) != r.n_micro:
        excluded["n_micro_mismatch"] += 1; continue
    if abs(sc0[r.default_microstate] - r.default_score) > 1e-9 or abs(sc0[r.best_microstate] - r.best_score) > 1e-9:
        excluded["score_mismatch"] += 1; continue
    runs = {0: dict(default=(bool(r.default_geom), bool(r.default_full_pb)), best=(bool(r.best_geom), bool(r.best_full_pb)), sc=dict(sc0))}
    miss = False
    for s in SEEDS:
        sck = dict(sc0); vd = {}
        for lab, mid in (("default", r.default_microstate), ("best", r.best_microstate)):
            v = bv.get("%s|%s|%d" % (r.uid, mid, s))
            if v is None or pd.isna(v[0]) or pd.isna(v[1]):
                miss = True; break
            vd[lab] = (bool(v[0]), bool(v[1]))
            sck[mid] = float(units[lab]["seeds"][str(s)]["score"])
        if miss:
            break
        runs[s] = dict(sc=sck, **vd)
    if miss:
        excluded["missing_verdict"] += 1; continue
    for lab, mid in (("default", r.default_microstate), ("best", r.best_microstate)):
        for c_i, c in enumerate(CRIT):
            v = []
            for k in [0] + SEEDS:
                run = runs[k]
                v.append(rank_pass(run["sc"], mid) if c == "score_rank" else run[lab][c_i])
            if c == "score_rank":
                orig = bool(getattr(r, lab + "_score_rank"))
                assert v[0] == orig, ("score_rank of run 0 differs from the decoy-matching ligand table", r.uid, lab)
            pairs_all = [v[i] != v[j] for i, j in itertools.combinations(range(5), 2)]
            pairs_orig = [v[0] != v[j] for j in range(1, 5)]
            rows.append(dict(set=r.set, target=r.target, role=r.role, uid=r.uid, unit=lab, criterion=c, verdicts="".join("1" if x else "0" for x in v), noise_flip_all_pairs=float(np.mean(pairs_all)), noise_flip_orig_vs_extra=float(np.mean(pairs_orig)),
                             score_sd=float(np.std([runs[k]["sc"][mid] for k in [0] + SEEDS], ddof=1))))
lu = pd.DataFrame(rows)
(RAW).mkdir(parents=True, exist_ok=True)
lu.to_csv(RAW / "noise_ligand_unit.csv", index=False)

rng = np.random.default_rng(BSEED)
out = []
used = lu[["set", "uid"]].drop_duplicates()
for (s, c), g in list(lu.groupby(["set", "criterion"])) + [(("ALL", c), lu[lu.criterion == c]) for c in CRIT]:
    flip = smp.set_index(["set", "uid"])["flip_" + c].astype(bool)
    keys = g[["set", "uid"]].drop_duplicates()
    lig = keys.merge(smp[["set", "uid", "flip_" + c]], on=["set", "uid"])
    k1, n1 = int(lig["flip_" + c].sum()), len(lig)
    per = g.groupby(["set", "uid"]).agg(noise_all=("noise_flip_all_pairs", "mean"), noise_orig=("noise_flip_orig_vs_extra", "mean")).reset_index().merge(lig, on=["set", "uid"])
    k2, n2 = float(g.noise_flip_all_pairs.sum()), len(g)
    z, p = F.two_proportion_z(k1, n1, k2, n2)
    d_obs = k1 / n1 - k2 / n2
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    f_ = per["flip_" + c].values.astype(float); nz_ = per.noise_all.values
    bs = f_[idx].mean(1) - nz_[idx].mean(1)
    out.append(dict(set=s, criterion=c, n_ligands=n1, n_ligand_units=n2, microstate_flip_k=k1, microstate_flip_rate=k1 / n1, microstate_flip_lo=F.wilson(k1, n1)[0], microstate_flip_hi=F.wilson(k1, n1)[1],
                    noise_flip_rate_all_pairs=k2 / n2, noise_flip_rate_orig_vs_extra=float(g.noise_flip_orig_vs_extra.mean()), diff=d_obs, diff_lo=float(np.percentile(bs, 2.5)), diff_hi=float(np.percentile(bs, 97.5)), z=z, p_value=p,
                    mean_score_sd_extra_seeds=float(g.score_sd.mean())))
res = pd.DataFrame(out)
fam = res[res.set != "ALL"].sort_values("p_value").reset_index()
m = len(fam); run_max = 0.0; adj = {}
for i, r_ in fam.iterrows():
    run_max = max(run_max, min(1.0, (m - i) * r_.p_value)) if not math.isnan(r_.p_value) else run_max
    adj[r_["index"]] = run_max
res["p_holm"] = [adj.get(i, float("nan")) for i in res.index]
path = RES / "noise_floor_vs_microstate_flip.csv"
res.to_csv(path, index=False)
desc = {"set": "ligand set (ALL = pooled, descriptive)", "criterion": "geom | full_pb | score_rank", "n_ligands": "ligands with complete data", "n_ligand_units": "(ligand, unit) pairs, unit = default or best microstate",
        "microstate_flip_k": "ligands whose default and best microstate differ in the criterion (original poses)", "microstate_flip_rate": "k / n_ligands", "microstate_flip_lo": "Wilson 95% lower", "microstate_flip_hi": "Wilson 95% upper",
        "noise_flip_rate_all_pairs": "mean over (ligand, unit) of the fraction of the 10 run pairs whose verdict differs", "noise_flip_rate_orig_vs_extra": "same, only the 4 original-vs-extra-seed pairs",
        "diff": "microstate_flip_rate - noise_flip_rate_all_pairs", "diff_lo": "ligand-cluster bootstrap 2.5%", "diff_hi": "ligand-cluster bootstrap 97.5%", "z": "pooled two-proportion z", "p_value": "two-sided",
        "mean_score_sd_extra_seeds": "mean SD of the Vina score over original + 4 extra seeds (kcal/mol)", "p_holm": "Holm over criteria x set (non-ALL rows)"}
P.write_schema(path, [(c, "", desc[c]) for c in res.columns], "one row per (set, criterion)", "Docking-noise flip rate versus microstate-induced flip rate",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [RES / "noise_sample.csv"], seed=BSEED, n_boot=N_BOOT, extra=dict(excluded=excluded, n_ligands_used=int(len(used))))))
rep_ok = [u for r_ in noise.values() for u in r_["units"] if u.get("status") == "ok"]
summ = dict(n_sampled=int(len(smp)), n_used=int(len(used)), excluded=excluded, n_noise_records=len(noise), n_bust_poses=int(len(bust)),
            conformer_reproduced_fraction=float(np.mean([u["conformer_reproduced"] for u in rep_ok])) if rep_ok else None, n_units_checked=len(rep_ok))
json.dump(summ, open(RES / "noise_floor_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1)); print(res.round(3).to_string(index=False))
