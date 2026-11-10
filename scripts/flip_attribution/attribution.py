import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pose_validity"))
import attribution_rule as A
import pose_flip as F

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

RDLogger.DisableLog("rdApp.*")
SRC = sys.argv[sys.argv.index("--source") + 1]
POSE_DIR = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
CRITERIA_DIR = Path(os.environ.get("CRITERIA_OUT_ROOT", str(ROOT)))
assert SRC in ("ten_target", "full_scale", "casf"), "--source must be ten_target, full_scale or casf"
OUT = Path(os.environ.get("ATTR_OUT_ROOT", str(ROOT)))
RES, RAW = OUT / "results/flip_attribution", OUT / "raw_outputs/flip_attribution"; RES.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
CRIT = F.CRITERIA
TYPES = ["protonation", "tautomer", "resonance", "stereo", "same_state", "unparsed"]
BSEED, PSEED, NB, NP = 20261051, 20261052, 2000, 10000
MIN_PAIRS = 20

if SRC == "ten_target":
    u = pd.read_parquet(POSE_DIR / "raw_outputs/pose_validity/ten_target_rep_pose_table.parquet")
    u["lig"] = u.dataset + "|" + u.seed.astype(str) + "|" + u.target + "|" + u.ligand_idx.astype(str)
    u["role"] = np.where(u.label.astype(int) == 1, "active", "negative")
    u["ligand_label"] = u.compound_id
    u["cluster"] = u.dataset + "|" + u.target + "|" + u.compound_id.astype(str)
    u["failed_checks"] = u.failed_checks.fillna("")
elif SRC == "casf":
    pc = pd.read_csv(CRITERIA_DIR / "results/pose_criteria/casf_pose_checks.csv", dtype={"pdbid": str, "microstate_id": str, "seed": str}, keep_default_na=False, na_values=[""])
    ms4 = pd.read_csv(ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv", dtype=str, keep_default_na=False)[["pdbid", "microstate_id", "smiles", "q"]]
    ms4 = ms4.rename(columns={"smiles": "stereo_tag", "q": "formal_charge"})
    pc = pc.rename(columns={"role": "state_role"})
    u = pc.merge(ms4, on=["pdbid", "microstate_id"], how="left", validate="m:1")
    assert u.formal_charge.notna().all() and u.stereo_tag.notna().all(), "a re-docked microstate is missing from the CASF-scoring table"
    u["formal_charge"] = u.formal_charge.astype(int)
    u["is_default"] = u.state_role.str.contains("single")
    u["dataset"] = "casf2016"; u["target"] = "casf2016"; u["role"] = "binder"
    u["lig"] = "casf2016|" + u.seed.astype(str) + "|" + u.pdbid
    u["ligand_label"] = u.pdbid
    u["cluster"] = u.pdbid
    u["fixedh_inchi"] = None
    u["failed_checks"] = u.failed_checks.fillna("")
else:
    u = pd.read_parquet(POSE_DIR / "raw_outputs/pose_validity/full_scale_pose_table.parquet")
    u = u.rename(columns={"set": "dataset"})
    u["lig"] = u.dataset + "|" + u.target + "|" + u.uid
    u["fixedh_inchi"] = None
    u["ligand_label"] = u.ligand_id.astype(str)
    u["cluster"] = u.lig
    u["failed_checks"] = u.failed_checks.fillna("")
for c in CRIT:
    u[c] = u[c].astype(bool)
u["vina_score"] = u.vina_score.astype(float)
u["is_default"] = u.is_default.astype(bool)
t0 = time.time()

best_idx = u.sort_values(["lig", "vina_score", "microstate_id"]).drop_duplicates("lig").set_index("lig").microstate_id
dflt = u[u.is_default].drop_duplicates("lig").set_index("lig")
oth = u[~u.is_default].copy()
oth = oth[oth.lig.isin(dflt.index)]
n_micro = u.groupby("lig").microstate_id.nunique()
heavy = {}


def n_heavy(smi):
    if smi not in heavy:
        m = Chem.MolFromSmiles(smi)
        heavy[smi] = m.GetNumHeavyAtoms() if m is not None else np.nan
    return heavy[smi]


pr = oth[["lig", "dataset", "target", "role", "ligand_label", "microstate_id", "formal_charge", "stereo_tag", "fixedh_inchi", "vina_score"] + CRIT + ["failed_checks"]].rename(
    columns={"microstate_id": "other_mid", "formal_charge": "q_other", "stereo_tag": "smi_other", "fixedh_inchi": "inchi_other", "vina_score": "score_other", "failed_checks": "failed_checks_other", **{c: c + "_other" for c in CRIT}})
d = dflt.reset_index()[["lig", "cluster", "microstate_id", "formal_charge", "stereo_tag", "fixedh_inchi", "vina_score", "failed_checks"] + CRIT].rename(
    columns={"microstate_id": "default_mid", "formal_charge": "q_default", "stereo_tag": "smi_default", "fixedh_inchi": "inchi_default", "vina_score": "score_default", "failed_checks": "failed_checks_default", **{c: c + "_default" for c in CRIT}})
pr = pr.merge(d, on="lig", how="inner")
pr["best_mid"] = pr.lig.map(best_idx)
pr["is_best_pair"] = (pr.other_mid == pr.best_mid) & (pr.best_mid != pr.default_mid)
pr["change_type"] = [A.change_type(q1, s1, q2, s2, i1, i2) for q1, s1, q2, s2, i1, i2 in zip(pr.q_default, pr.smi_default, pr.q_other, pr.smi_other, pr.inchi_default, pr.inchi_other)]
pr["change_type_fine"] = [A.change_type_fine(q1, s1, q2, s2, c) for q1, s1, q2, s2, c in zip(pr.q_default, pr.smi_default, pr.q_other, pr.smi_other, pr.change_type)]
pr["charge_delta"] = pr.q_other.astype(int) - pr.q_default.astype(int)
pr["n_heavy"] = [n_heavy(s) for s in pr.smi_default]
pr["n_micro"] = pr.lig.map(n_micro)
pr["gain"] = pr.score_default - pr.score_other
for c in CRIT:
    pr["flip_" + c] = pr[c + "_default"] != pr[c + "_other"]
    pr["rescue_" + c] = (~pr[c + "_default"]) & pr[c + "_other"]
    pr["degrade_" + c] = pr[c + "_default"] & ~pr[c + "_other"]
pr.drop(columns=["inchi_default", "inchi_other"]).to_parquet(RAW / ("pairs_%s.parquet" % SRC))
print("[%.0fs] %s: %d ligands with a default, %d pairs, types %s" % (time.time() - t0, SRC, pr.lig.nunique(), len(pr), pr.change_type.value_counts().to_dict()), flush=True)

noise = {}
nf = POSE_DIR / "results/pose_validity/noise_floor_vs_microstate_flip.csv"
if nf.exists():
    nz = pd.read_csv(nf)
    for r in nz.itertuples():
        noise[(r.set, r.criterion)] = (r.noise_flip_rate_all_pairs, r.n_ligand_units)
if SRC == "casf":
    nf7 = CRITERIA_DIR / "results/pose_criteria/casf_noise_floor.csv"
    if nf7.exists():
        for r in pd.read_csv(nf7).itertuples():
            if r.criterion in CRIT:
                noise[("casf2016", r.criterion)] = (r.noise_flip_rate, r.n_complexes)


def table(scope):
    base = pr[pr.is_best_pair] if scope == "best_pair" else pr
    rows = []
    groups = [(ds, t, g) for (ds, t), g in base.groupby(["dataset", "target"])] + [(ds, "ALL", g) for ds, g in base.groupby("dataset")]
    for ds, t, g in groups:
        for c in CRIT:
            tot_flip = int(g["flip_" + c].sum())
            for ty in TYPES:
                gg = g[g.change_type == ty]
                n, k = len(gg), int(gg["flip_" + c].sum())
                if n == 0:
                    continue
                lo, hi = F.wilson(k, n)
                row = dict(source=SRC, dataset=ds, target=t, scope=scope, criterion=c, change_type=ty, n_total_pairs_of_type=n, n_flips=k, flip_rate_within_type=k / n, wilson_lo=lo, wilson_hi=hi,
                           share_of_flips=(k / tot_flip) if tot_flip else np.nan, n_rescue=int(gg["rescue_" + c].sum()), n_degrade=int(gg["degrade_" + c].sum()), share_of_pairs=n / len(g))
                if scope == "all_pairs" and t == "ALL":
                    key = (ds, c) if (ds, c) in noise else (("ALL", c) if SRC == "full_scale" and ("ALL", c) in noise else None)
                    if key:
                        r0, n0 = noise[key]
                        z, p = F.two_proportion_z(k, n, r0 * n0, n0)
                        row.update(noise_key=key[0], noise_rate=r0, noise_n=n0, diff_vs_noise=k / n - r0, z_vs_noise=z, p_vs_noise=p)
                rows.append(row)
    return pd.DataFrame(rows)


def cluster_ci(tab):
    rng = np.random.default_rng(BSEED)
    cil, cih = [], []
    for r in tab.itertuples():
        if not (r.scope == "all_pairs" and r.target == "ALL"):
            cil.append(np.nan); cih.append(np.nan); continue
        g = pr[(pr.dataset == r.dataset) & (pr.change_type == r.change_type)]
        s = g.groupby("cluster")["flip_" + r.criterion].agg(["sum", "count"]).values
        idx = rng.integers(0, len(s), size=(NB, len(s)))
        tot = s[idx].sum(1)
        rates = tot[:, 0] / tot[:, 1]
        cil.append(float(np.percentile(rates, 2.5))); cih.append(float(np.percentile(rates, 97.5)))
    tab["cluster_ci_lo"], tab["cluster_ci_hi"] = cil, cih
    return tab


att = pd.concat([table("best_pair"), table("all_pairs")], ignore_index=True)
att = cluster_ci(att)
if "p_vs_noise" in att.columns:
    m = att.p_vs_noise.notna()
    pv = att.loc[m, "p_vs_noise"].values; order = np.argsort(pv); run = 0.0; hp = np.empty(len(pv))
    for rank, i in enumerate(order):
        run = max(run, (len(pv) - rank) * pv[i]); hp[i] = min(1.0, run)
    att.loc[m, "p_holm_vs_noise"] = hp
path = RES / ("flip_attribution_%s.csv" % SRC)
att.to_csv(path, index=False)

fine_rows, chk_rows = [], []
for scope, base in (("best_pair", pr[pr.is_best_pair]), ("all_pairs", pr)):
    for ds, g in base.groupby("dataset"):
        for c in CRIT:
            for ty, gg in g.groupby("change_type_fine"):
                k = int(gg["flip_" + c].sum()); lo, hi = F.wilson(k, len(gg))
                fine_rows.append(dict(source=SRC, dataset=ds, scope=scope, criterion=c, change_type_fine=ty, n_total_pairs_of_type=len(gg), n_flips=k, flip_rate_within_type=k / len(gg), wilson_lo=lo, wilson_hi=hi,
                                      share_of_flips=k / max(int(g["flip_" + c].sum()), 1), share_of_pairs=len(gg) / len(g)))
for ds, g in pr[pr.is_best_pair & pr.flip_full_pb].groupby("dataset"):
    for ty, gg in g.groupby("change_type_fine"):
        for direction, sub, col in (("rescue (default invalid -> best valid)", gg[gg.rescue_full_pb], "failed_checks_default"), ("degrade (default valid -> best invalid)", gg[gg.degrade_full_pb], "failed_checks_other")):
            cnt = pd.Series([x for s_ in sub[col].fillna("") for x in s_.split(";") if x]).value_counts()
            for chk_name, n in cnt.items():
                chk_rows.append(dict(source=SRC, dataset=ds, change_type_fine=ty, direction=direction, n_flips_of_type_direction=len(sub), failed_check=chk_name, n_flips_with_check=int(n), share=n / len(sub)))
pd.DataFrame(fine_rows).to_csv(RES / ("flip_attribution_fine_%s.csv" % SRC), index=False)
pd.DataFrame(chk_rows).to_csv(RES / ("flip_failed_checks_%s.csv" % SRC), index=False)
print("[%.0fs] attribution table %s" % (time.time() - t0, att.shape), flush=True)

import statsmodels.api as sm
import statsmodels.formula.api as smf

logi = {}
for ds, gd in pr.groupby("dataset"):
    gd = gd[gd.change_type.isin(["protonation", "tautomer", "stereo", "resonance"])].copy()
    cnt = gd.change_type.value_counts()
    keep = [t for t in cnt.index if cnt[t] >= MIN_PAIRS]
    gd = gd[gd.change_type.isin(keep)].dropna(subset=["n_heavy"]).copy()
    gd["z_heavy"] = (gd.n_heavy - gd.n_heavy.mean()) / gd.n_heavy.std()
    gd["log_nmicro"] = np.log(gd.n_micro)
    gd["is_prot"] = (gd.change_type == "protonation").astype(int)
    for c in CRIT:
        gd["y"] = gd["flip_" + c].astype(int)
        ev = gd.groupby("target").y.agg(["sum", "count"])
        drop_t = ev.index[(ev["sum"] == 0) | (ev["sum"] == ev["count"])].tolist()
        gc = gd[~gd.target.isin(drop_t)].copy()
        rec = dict(n_pairs=len(gc), n_ligands=gc.lig.nunique(), n_clusters=gc.cluster.nunique(), change_types_kept=keep, pair_counts={k: int(v) for k, v in gc.change_type.value_counts().items()}, n_flips=int(gc.y.sum()), targets_dropped_no_variation=drop_t)
        for name, form in (("types_vs_tautomer", "y ~ C(change_type, Treatment('tautomer')) + C(target) + z_heavy + log_nmicro"), ("protonation_vs_rest", "y ~ is_prot + C(target) + z_heavy + log_nmicro")):
            try:
                if gc.y.sum() < 5 or gc.y.sum() > len(gc) - 5 or ("tautomer" not in keep and name == "types_vs_tautomer"):
                    raise ValueError("too few events / no reference class")
                res = smf.glm(form if gc.target.nunique() > 1 else form.replace(" + C(target)", ""), data=gc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(gc.cluster)[0]})
                if not (np.isfinite(res.params).all() and np.isfinite(res.bse).all()):
                    raise ValueError("non-finite coefficient or standard error (separation)")
                ci = res.conf_int()
                rec[name] = {k: dict(coef=float(res.params[k]), odds_ratio=float(np.exp(res.params[k])), or_lo=float(np.exp(ci.loc[k, 0])), or_hi=float(np.exp(ci.loc[k, 1])), p=float(res.pvalues[k]))
                             for k in res.params.index if k.startswith("C(change_type") or k in ("is_prot", "z_heavy", "log_nmicro")}
            except Exception as exc:
                rec[name] = dict(error=str(exc)[:200])
        logi["%s|%s" % (ds, c)] = rec
json.dump(dict(source=SRC, formula_note="logistic regression (binomial GLM) with cluster-robust standard errors (clusters = compounds; the 5 seeds of one compound form one cluster); reference class tautomer; covariates: target fixed effects, z-scored heavy atoms of the default microstate, log(n microstates); classes with < %d pairs and targets without outcome variation dropped. A GEE with exchangeable working correlation was tried first on a subset and returned NaN for score_rank (highly correlated within-ligand outcomes) -- replaced for all criteria before any result existed" % MIN_PAIRS, results=logi,
               provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [], seed=BSEED)), open(RES / ("flip_logistic_%s.json" % SRC), "w"), indent=2, ensure_ascii=False)
print("[%.0fs] logistic done" % (time.time() - t0), flush=True)

rng = np.random.default_rng(PSEED)
jak = {}
for ds, gd in pr.groupby("dataset"):
    if "jak2" not in set(gd.target):
        continue
    for c in CRIT:
        bp = gd[gd.is_best_pair & gd["flip_" + c]]
        rec = dict(n_flips_jak2=int((bp.target == "jak2").sum()), n_flips_rest=int((bp.target != "jak2").sum()))
        types_present = sorted(bp.change_type.unique())
        if rec["n_flips_jak2"] and rec["n_flips_rest"] and len(types_present) >= 2:
            M = pd.crosstab(bp.cluster, bp.change_type).reindex(columns=types_present, fill_value=0)
            lab = (bp.drop_duplicates("cluster").set_index("cluster").target.reindex(M.index) == "jak2").values
            X = M.values.astype(float)

            def chi2_of(l):
                t = np.vstack([X[l].sum(0), X[~l].sum(0)]); t = t[:, t.sum(0) > 0]
                return stats.chi2_contingency(t, correction=False)[0] if t.shape[1] >= 2 else 0.0
            obs = chi2_of(lab)
            cnt = sum(chi2_of(rng.permutation(lab)) >= obs - 1e-12 for _ in range(NP))
            rec.update(chi2=float(obs), perm_p=(cnt + 1) / (NP + 1), permutation_unit="compound cluster (target labels permuted across the clusters that have a flip)",
                       jak2_type_counts=bp[bp.target == "jak2"].change_type.value_counts().to_dict(), rest_type_counts=bp[bp.target != "jak2"].change_type.value_counts().to_dict(),
                       jak2_share_protonation=float((bp[bp.target == "jak2"].change_type == "protonation").mean()), rest_share_protonation=float((bp[bp.target != "jak2"].change_type == "protonation").mean()))
        else:
            rec["note"] = "contingency table degenerate (one class or no JAK2 flips)"
        bpl = gd[gd.is_best_pair].groupby("target")["lig"].nunique()
        nlig = gd.groupby("target")["lig"].nunique()
        fl = gd[gd.is_best_pair & gd["flip_" + c]].groupby("target")["lig"].nunique().reindex(nlig.index).fillna(0)
        pct = (100 * fl / nlig).sort_values(ascending=False)
        rec.update(flip_pct_by_target={k: float(v) for k, v in pct.items()}, jak2_flip_pct=float(pct.get("jak2", np.nan)), jak2_rank=int(list(pct.index).index("jak2") + 1) if "jak2" in pct.index else None, n_targets=len(pct),
                   jak2_percentile=float(100 * (pct < pct.get("jak2", np.nan)).mean()) if "jak2" in pct.index else None)
        jak["%s|%s" % (ds, c)] = rec
resc = pr[pr.is_best_pair & pr.rescue_full_pb & (pr.change_type == "protonation")].copy()
resc = resc.sort_values("gain", ascending=False).reset_index(drop=True)
resc["gain_rank"] = np.arange(1, len(resc) + 1)
resc["gain_percentile"] = 100 * (1 - (resc.gain_rank - 1) / max(len(resc), 1))
prov_cids = set()
pj = Path("$BASE_ROOT/results/benchmark/provenance_case/jak2/provenance_flips.json")
if SRC == "ten_target" and pj.exists():
    prov_cids = {f["cid"] for f in json.load(open(pj))["flips"]}
resc["in_manuscript_jak2_provenance_flips"] = resc.ligand_label.isin(prov_cids) & (resc.target == "jak2")
cat_cols = ["dataset", "target", "role", "ligand_label", "default_mid", "other_mid", "charge_delta", "failed_checks_default", "gain", "gain_rank", "gain_percentile", "in_manuscript_jak2_provenance_flips"]
resc = resc.rename(columns={"other_mid": "other_mid"})[cat_cols + ["score_default", "score_other"]].rename(columns={"ligand_label": "ligand_id", "default_mid": "default_microstate", "other_mid": "best_microstate"})
resc.to_csv(RES / ("rescue_case_catalog_%s.csv" % SRC), index=False)
jak["rescue_catalog"] = dict(n_cases=len(resc), n_jak2=int((resc.target == "jak2").sum()), jak2_cases=resc[resc.target == "jak2"][["ligand_id", "gain", "gain_rank", "gain_percentile", "in_manuscript_jak2_provenance_flips"]].to_dict("records"),
                             by_target=resc.target.value_counts().to_dict())
json.dump(dict(source=SRC, results=jak, provenance=P.provenance_block(Path(__file__), ROOT / "configs/pose_validity.yaml", [], seed=PSEED, n_perm=NP)), open(RES / ("jak2_representativeness_%s.json" % SRC), "w"), indent=2, ensure_ascii=False, default=str)
print("[%.0fs] done | rescue cases %d (jak2 %d)" % (time.time() - t0, len(resc), int((resc.target == "jak2").sum())), flush=True)
