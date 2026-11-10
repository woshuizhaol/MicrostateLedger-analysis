import json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pose_validity"))
import pose_flip as F

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/pose_validity.yaml", encoding="utf-8"))
OUT = Path(os.environ.get("CRITERIA_OUT_ROOT", str(ROOT)))
CHK, RES = OUT / "raw_outputs/pose_criteria/checks", OUT / "results/pose_criteria"; RES.mkdir(parents=True, exist_ok=True)
CRIT = ["geom", "full_pb", "score_rank", "rmsd_ref", "plif"]
RMSD_T, PLIF_T = float(CFG["criteria"]["rmsd_threshold_A"]), float(CFG["criteria"]["plif_threshold"])
NB, BSEED = 2000, 20261061
t0 = time.time()

ms = pd.read_csv(ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv", dtype=str, keep_default_na=False)
single = ms[ms.roles.str.split(";").map(lambda r: "single" in r)].set_index("pdbid").microstate_id
dom = ms[ms.roles.str.split(";").map(lambda r: "dominant" in r)].set_index("pdbid").microstate_id
assert single.index.is_unique and dom.index.is_unique, "a complex has two single / dominant microstates"
parts = sorted(CHK.glob("*.parquet"))
d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
for c in ("geom", "full_pb", "inter"):
    d[c] = d[c].astype(bool)
d["role"] = [("single" if m == single.get(p) else "") + ("+" if m == single.get(p) and m == dom.get(p) else "") + ("dominant" if m == dom.get(p) else "") or "other" for p, m in zip(d.pdbid, d.microstate_id)]
d["score_rank"] = False
for _, ix in d.groupby(["pdbid", "seed"]).indices.items():
    d.iloc[ix, d.columns.get_loc("score_rank")] = F.score_rank_pass(d.iloc[ix]["vina_score"].values)
d["rmsd_ref"] = pd.array([pd.NA if (isinstance(x, float) and math.isnan(x)) else bool(x <= RMSD_T) for x in d.rmsd], dtype="boolean")
d["plif"] = pd.array([pd.NA if (isinstance(x, float) and math.isnan(x)) else bool(x >= PLIF_T) for x in d.plif_tanimoto], dtype="boolean")
print("[%.0fs] %d complexes, %d poses" % (time.time() - t0, d.pdbid.nunique(), len(d)), flush=True)

pt = d[["pdbid", "microstate_id", "seed", "role", "vina_score", "conf_rank", "geom", "full_pb", "score_rank", "rmsd", "rmsd_ref", "plif_tanimoto", "plif", "plif_receptor_route", "pb_timeout", "failed_checks", "rmsd_reason", "plif_reason"]].copy()
for c in ("rmsd_ref", "plif"):
    pt[c] = pt[c].astype(object).where(pt[c].notna(), "")
pt_path = RES / "casf_pose_checks.csv"
pt.to_csv(pt_path, index=False, float_format="%.6g")
cols = {"pdbid": "CASF-2016 complex", "microstate_id": "microstate id of raw_outputs/casf_scoring/per_microstate_inplace_scores.csv", "seed": "Vina seed", "role": "single | dominant | single+dominant | other", "vina_score": "top-1 mode score (kcal/mol)",
        "conf_rank": "conformer used (0 = lowest energy)", "geom": "PoseBusters intramolecular checks all True", "full_pb": "all 22 PoseBusters checks True", "score_rank": "rank <= ceil(n/2) among the complex's microstates of the same seed",
        "rmsd": "symmetry-corrected heavy-atom RMSD to the crystal ligand (A)", "rmsd_ref": "rmsd <= 2.0 A (empty = NA)", "plif_tanimoto": "ProLIF Tanimoto to the crystal fingerprint", "plif": "Tanimoto >= 0.5 (empty = NA)",
        "plif_receptor_route": "how the receptor of the fingerprint was built (rdkit | rdkit_nopropcheck | none)", "pb_timeout": "True when the PoseBusters energy-ratio check hit the CPU guard (internal_energy = NaN, counted as failed); use to exclude these poses in a sensitivity analysis", "failed_checks": "failed PoseBusters columns", "rmsd_reason": "reason when rmsd is NA", "plif_reason": "reason when plif_tanimoto is NA"}
P.write_schema(pt_path, [(c, "", cols[c]) for c in pt.columns], "one row per (complex, microstate, seed) pose", "Pose criteria of the re-docked CASF-2016 poses",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv"], seed=BSEED, n_boot=NB, extra=dict(n_checks_files=len(parts)))))

aff = sorted(p for p in single.index if p in dom.index and single[p] != dom[p] and p in set(d.pdbid))
s_ = d[d.role.str.contains("single")][["pdbid", "seed"] + CRIT].rename(columns={c: c + "_s" for c in CRIT})
m_ = d[d.role.str.contains("dominant")][["pdbid", "seed"] + CRIT].rename(columns={c: c + "_d" for c in CRIT})
pair = s_.merge(m_, on=["pdbid", "seed"])
pair = pair[pair.pdbid.isin(aff)]
rng = np.random.default_rng(BSEED)
rows, cx = [], {}
for c in CRIT:
    a, b = pair[c + "_s"], pair[c + "_d"]
    ok = a.notna() & b.notna()
    q = pair[ok].assign(a=a[ok].astype(bool), b=b[ok].astype(bool))
    q["flip"] = q.a != q.b; q["rescue"] = ~q.a & q.b; q["degrade"] = q.a & ~q.b
    per = q.groupby("pdbid").agg(flip=("flip", "mean"), n=("flip", "size"))
    cx[c] = per.flip
    idx = rng.integers(0, len(per), size=(NB, len(per))) if len(per) else None
    bs = per.flip.values[idx].mean(1) if len(per) else np.array([np.nan])
    rows.append(dict(criterion=c, scope="all_seeds", seed="ALL", n_complexes=len(per), n_pairs=len(q), n_flip=int(q.flip.sum()), flip_pct=100 * float(per.flip.mean()) if len(per) else np.nan,
                     ci_lo=100 * float(np.percentile(bs, 2.5)), ci_hi=100 * float(np.percentile(bs, 97.5)), n_rescue=int(q.rescue.sum()), n_degrade=int(q.degrade.sum()), n_complexes_with_any_flip=int((per.flip > 0).sum())))
    for s, g in q.groupby("seed"):
        rows.append(dict(criterion=c, scope="one_seed", seed=int(s), n_complexes=g.pdbid.nunique(), n_pairs=len(g), n_flip=int(g.flip.sum()), flip_pct=100 * float(g.flip.mean()), ci_lo=100 * F.wilson(int(g.flip.sum()), len(g))[0],
                         ci_hi=100 * F.wilson(int(g.flip.sum()), len(g))[1], n_rescue=int(g.rescue.sum()), n_degrade=int(g.degrade.sum()), n_complexes_with_any_flip=int(g.flip.sum())))
fl = pd.DataFrame(rows)
fl_path = RES / "casf_flip_by_criterion.csv"; fl.to_csv(fl_path, index=False, float_format="%.6g")
fcols = {"criterion": "geom | full_pb | score_rank | rmsd_ref | plif", "scope": "all_seeds (complex-level mean, bootstrap CI) | one_seed (pooled complexes of that seed, Wilson CI)", "seed": "Vina seed or ALL",
         "n_complexes": "affected complexes with both verdicts", "n_pairs": "(complex, seed) pairs with both verdicts", "n_flip": "pairs with different verdicts", "flip_pct": "flip percentage (see scope)", "ci_lo": "95% lower", "ci_hi": "95% upper",
         "n_rescue": "single invalid -> dominant valid", "n_degrade": "single valid -> dominant invalid", "n_complexes_with_any_flip": "complexes (all_seeds) / pairs (one_seed) with a flip"}
P.write_schema(fl_path, [(c, "%", fcols[c]) if c in ("flip_pct", "ci_lo", "ci_hi") else (c, "", fcols[c]) for c in fl.columns], "one row per criterion x scope", "Validity flip rate between the default and the dominant microstate on re-docked CASF-2016",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [pt_path], seed=BSEED, n_boot=NB, extra=dict(n_affected=len(aff)))))

nrows = []
for c in CRIT:
    per_c = {}
    for p in aff:
        vals = []
        for role_mid in (single[p], dom[p]):
            g = d[(d.pdbid == p) & (d.microstate_id == role_mid)].sort_values("seed")[c].dropna().astype(bool).values
            if len(g) >= 2:
                prs = [g[i] != g[j] for i in range(len(g)) for j in range(i + 1, len(g))]
                vals.append(float(np.mean(prs)))
        if vals:
            per_c[p] = float(np.mean(vals))
    noise = pd.Series(per_c); flip = cx[c]
    j = pd.concat([flip.rename("flip"), noise.rename("noise")], axis=1).dropna()
    diff = (j.flip - j.noise).values
    idx = rng.integers(0, len(j), size=(NB, len(j))) if len(j) else None
    bs = diff[idx].mean(1) if len(j) else np.array([np.nan])
    try:
        w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided") if np.any(diff != 0) else None
        pw = float(w.pvalue) if w is not None else 1.0
    except ValueError:
        pw = 1.0
    k1, n1 = float(j.flip.sum()), len(j)
    z, pz = F.two_proportion_z(k1, n1, float(j.noise.sum()), len(j))
    nrows.append(dict(criterion=c, n_complexes=len(j), microstate_flip_rate=float(j.flip.mean()), noise_flip_rate=float(j.noise.mean()), diff=float(diff.mean()) if len(j) else np.nan,
                      diff_lo=float(np.percentile(bs, 2.5)), diff_hi=float(np.percentile(bs, 97.5)), p_wilcoxon_paired=pw, z_naive=z, p_naive=pz))
nz = pd.DataFrame(nrows)
order = np.argsort(nz.p_wilcoxon_paired.values); run = 0.0; hp = np.empty(len(nz))
for r_, i in enumerate(order):
    run = max(run, (len(nz) - r_) * nz.p_wilcoxon_paired.values[i]); hp[i] = min(1.0, run)
nz["p_holm"] = hp
nz_path = RES / "casf_noise_floor.csv"; nz.to_csv(nz_path, index=False, float_format="%.6g")
ncols = {"criterion": "criterion", "n_complexes": "affected complexes with a flip fraction and a noise estimate", "microstate_flip_rate": "mean over complexes of the single-vs-dominant flip fraction (over seeds)",
         "noise_flip_rate": "mean over complexes of the seed-pair disagreement of the SAME microstate (single and dominant averaged)", "diff": "microstate_flip_rate - noise_flip_rate (paired mean)", "diff_lo": "complex bootstrap 2.5%", "diff_hi": "complex bootstrap 97.5%",
         "p_wilcoxon_paired": "Wilcoxon signed-rank on the paired complex differences (two-sided)", "z_naive": "pooled two-proportion z (units treated as independent; descriptive only)", "p_naive": "p of z_naive", "p_holm": "Holm over the 5 criteria of p_wilcoxon_paired"}
P.write_schema(nz_path, [(c, "", ncols[c]) for c in nz.columns], "one row per criterion", "Docking-noise floor versus microstate-induced validity flip on CASF-2016",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [pt_path], seed=BSEED, n_boot=NB)))

krows = []
pairs = [("geom", "full_pb"), ("geom", "score_rank"), ("full_pb", "score_rank")] + [(a, b) for a in ("geom", "full_pb", "score_rank") for b in ("rmsd_ref", "plif")] + [("rmsd_ref", "plif")]
for x, y in pairs:
    q = d[d[x].notna() & d[y].notna()].copy()
    q[x] = q[x].astype(bool); q[y] = q[y].astype(bool)
    r = F.kappa_cluster(q.assign(lig=q.pdbid), x, y, ["lig"], NB, rng)
    krows.append(dict(criterion_a=x, criterion_b=y, **r))
kp = pd.DataFrame(krows)
kp_path = RES / "casf_criteria_kappa.csv"; kp.to_csv(kp_path, index=False, float_format="%.6g")
kcols = {c: "see pose_flip.kappa_cluster" for c in kp.columns}
kcols.update(criterion_a="first criterion", criterion_b="second criterion", n_poses="poses with both verdicts", n_clusters="complexes", kappa="Cohen's kappa", kappa_lo="complex bootstrap 2.5%", kappa_hi="complex bootstrap 97.5%",
             p_both_true="share of poses valid under both", p_first_true="share valid under the first", p_second_true="share valid under the second", agree="observed agreement")
P.write_schema(kp_path, [(c, "", kcols[c]) for c in kp.columns], "one row per criterion pair", "Pairwise Cohen's kappa of the pose criteria on re-docked CASF-2016 poses",
               extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [pt_path], seed=BSEED, n_boot=NB)))

summ = dict(n_complexes_redocked=int(d.pdbid.nunique()), n_poses=len(d), n_affected_complexes_with_poses=len(aff), pass_rate={c: float(d[c].dropna().astype(bool).mean()) for c in CRIT}, n_na={c: int(d[c].isna().sum()) for c in CRIT},
            rmsd_reasons=d.loc[d.rmsd.isna(), "rmsd_reason"].value_counts().to_dict(), plif_reasons=d.loc[d.plif_tanimoto.isna(), "plif_reason"].value_counts().to_dict(), plif_route_counts=d.drop_duplicates("pdbid").plif_receptor_route.value_counts().to_dict(), plif_na_complexes=sorted(d[d.plif_tanimoto.isna()].pdbid.unique().tolist()),
            n_affected_missing_single_or_dominant=int(sum(1 for p in single.index if p in set(d.pdbid) and single[p] != dom.get(p, single[p]) and not ((d.pdbid == p) & (d.microstate_id == dom[p])).any())), seconds=round(time.time() - t0, 1))
json.dump(summ, open(RES / "casf_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1)); print(fl[fl.scope == "all_seeds"].round(2).to_string(index=False)); print(nz.round(4).to_string(index=False)); print(kp.round(3).to_string(index=False))
