import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decoy_stats as S

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
T = CFG["tests"]
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RES = OUT_ROOT / "results/decoy_matching"
SETS = ["dude_full", "litpcba", "zinc_unmatched", "zinc_charge_mismatch"]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); out = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); out[i] = min(1.0, run)
    return out


def load(set_name):
    df = pd.read_csv(RES / ("%s_per_ligand.csv" % set_name), dtype={"ligand_id": str, "charge_class": str, "sites_class": str}, keep_default_na=False, na_values=[""])
    df["is_active"] = (df["role"] == "active").astype(int)
    return df


def battery(df, gaincol, seed, n_boot=None, n_hl=None, n_perm=None, full=True):
    n_boot = T["n_boot_delta"] if n_boot is None else n_boot
    n_hl = T["n_boot_hl"] if n_hl is None else n_hl
    n_perm = T["n_perm"] if n_perm is None else n_perm
    d = df.dropna(subset=[gaincol]).reset_index(drop=True)
    g = d[gaincol].values.astype(float); act = d["is_active"].values.astype(bool)
    a, b = g[act], g[~act]
    out = dict(gaincol=gaincol, n_active=int(act.sum()), n_negative=int((~act).sum()), median_active=float(np.median(a)), median_negative=float(np.median(b)), mean_active=float(a.mean()), mean_negative=float(b.mean()),
               frac_zero_active=float(np.mean(a == 0)), frac_zero_negative=float(np.mean(b == 0)))
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    out["mann_whitney"] = dict(U=float(mw.statistic), p=float(mw.pvalue))
    delta = S.cliffs_delta(a, b); hl = S.hl_shift(a, b)
    rng = np.random.default_rng(T["seed_boot"] + seed)
    bd, bh = S.bootstrap_effects(d, gaincol, n_boot, rng, n_hl=n_hl)
    out["cliffs_delta"] = dict(estimate=float(delta), ci95=[float(x) for x in np.percentile(bd, [2.5, 97.5])], tost=S.equivalence(bd, T["cliffs_delta_equiv_bound"], T["alpha"]), bound=T["cliffs_delta_equiv_bound"])
    out["hodges_lehmann"] = dict(estimate=float(hl), ci95=[float(x) for x in np.percentile(bh, [2.5, 97.5])], tost=S.equivalence(bh, T["hl_equiv_bound_kcal"], T["alpha"]), bound=T["hl_equiv_bound_kcal"], n_boot=n_hl)
    if not full:
        return out, None
    s, cuts = S.build_strata(d)
    ve, tab = S.van_elteren(g, act, s.values)
    sd = S.stratified_delta(tab)
    valid = ~s.isna().values
    sub = d[valid].reset_index(drop=True); ss = s[valid].reset_index(drop=True)
    codes = pd.factorize(ss)[0]; ncodes = int(codes.max()) + 1 if len(codes) else 0
    gs, as_ = sub[gaincol].values.astype(float), sub["is_active"].values.astype(bool)
    cells = S.cell_index(sub)
    rng2 = np.random.default_rng(T["seed_boot"] + 1000 + seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ix = S.boot_draws(cells, rng2)
        bs[i] = S.strat_delta_fast(gs[ix], as_[ix], codes[ix], ncodes)
    perm_p, T0 = S.permutation_pvalue(gs, as_, ss.values, n_perm, np.random.default_rng(T["seed_perm"] + seed))
    out["stratified"] = dict(van_elteren=ve, stratified_delta=float(sd), stratified_delta_ci95=[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])], tost=S.equivalence(bs[np.isfinite(bs)], T["cliffs_delta_equiv_bound"], T["alpha"]),
                             permutation_p=perm_p, n_strata_total=int(tab.shape[0]), n_strata_used=int(tab["used"].sum()), n_ligands_in_strata=int(valid.sum()), n_ligands_excluded_no_property=int((~valid).sum()), quartile_cuts_n_eligible=[float(x) for x in cuts])
    mm = S.mixed_model(d, gaincol); mmr = S.mixed_model_rank(d, gaincol)
    out["mixed_model"] = mm; out["mixed_model_rank"] = mmr
    return out, tab


def vs_metrics(df, set_name, seed):
    d = df[(df["default_docked"] == True) & (df["in_prefix"] == True)].copy()
    rows = []
    for t, g in d.groupby("target"):
        y = g["is_active"].values
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        for strat, col in (("default", "score_default"), ("full", "score_first4_min"), ("oracle", "score_min")):
            s = g[col].values.astype(float)
            rows.append(dict(set=set_name, target=t, strategy=strat, n_active=int(y.sum()), n_negative=int((1 - y).sum()), auc=S.auc(y, s), ef1=S.ef(y, s, 1.0), ef5=S.ef(y, s, 5.0), bedroc20=S.bedroc(y, s),
                             median_score_active=float(np.median(s[y == 1])), median_score_negative=float(np.median(s[y == 0]))))
    tab = pd.DataFrame(rows)
    summ = {}
    if len(tab):
        for st in ("default", "full", "oracle"):
            x = tab[tab.strategy == st]
            summ[st] = {m: float(x[m].mean()) for m in ("auc", "ef1", "ef5", "bedroc20")}
        piv = tab.pivot(index="target", columns="strategy", values="auc")
        tg = piv.index.values
        rng = np.random.default_rng(T["seed_boot"] + 5000 + seed)
        for a_, b_ in (("oracle", "default"), ("full", "default"), ("oracle", "full")):
            diff = (piv[a_] - piv[b_]).values
            bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(T["n_boot_delta"])])
            summ["dAUC_%s_minus_%s" % (a_, b_)] = dict(mean=float(diff.mean()), ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])], per_target=dict(zip(tg.tolist(), [float(v) for v in diff])), n_targets=len(diff))
        sig = tab.pivot(index="target", columns="strategy", values="median_score_active") - tab.pivot(index="target", columns="strategy", values="median_score_negative")
        summ["discrimination_signal_median_diff"] = {st: float(sig[st].mean()) for st in ("default", "full", "oracle")}
    return tab, summ


def main():
    t0 = time.time()
    present = [s for s in SETS if (RES / ("%s_per_ligand.csv" % s)).exists()]
    results, vs_all = {}, {}
    for k, s in enumerate(present):
        df = load(s)
        use = df[df["in_analysis"] == True].copy()
        res = dict(set=s, n_rows=len(df), n_in_analysis=len(use), rounds_in_analysis=sorted(use["round"].unique().tolist()))
        for col in ("gain", "gain_first4"):
            r, tab = battery(use, col, seed=k)
            res[col] = r
            if col == "gain" and tab is not None:
                tab.to_csv(RES / ("gain_stratified_%s.csv" % s), index=False)
        use2 = df[df["in_analysis_incl_single"] == True].copy()
        res["n_in_analysis_incl_single"] = len(use2)
        res["gain_incl_single"], _ = battery(use2, "gain", seed=k + 50)
        rows = []
        for tg, g in use.groupby("target"):
            a, b = g[g.is_active == 1]["gain"].values, g[g.is_active == 0]["gain"].values
            if len(a) and len(b):
                rows.append(dict(set=s, target=tg, n_active=len(a), n_negative=len(b), median_active=float(np.median(a)), median_negative=float(np.median(b)), cliffs_delta=S.cliffs_delta(a, b), mw_p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)))
        pd.DataFrame(rows).to_csv(RES / ("gain_by_target_%s.csv" % s), index=False)
        json.dump(dict(set=s, primary=res["gain"]["mixed_model"], rank_sensitivity=res["gain"]["mixed_model_rank"], secondary=res["gain_first4"]["mixed_model"]), open(RES / ("gain_mixedlm_%s.json" % s), "w"), indent=2, ensure_ascii=False)
        json.dump(res, open(RES / ("gain_tests_%s.json" % s), "w"), indent=2, ensure_ascii=False)
        vt, vsum = vs_metrics(df, s, k)
        vt.to_csv(RES / ("vs_metrics_%s.csv" % s), index=False); vs_all[s] = vsum
        results[s] = res
        print("[%.0fs] %s: n=%d (%d actives) | MW p=%.3g delta=%.3f [%.3f, %.3f] | VE p=%.3g | mixed coef=%.3f" % (time.time() - t0, s, res["n_in_analysis"], res["gain"]["n_active"], res["gain"]["mann_whitney"]["p"], res["gain"]["cliffs_delta"]["estimate"],
              *res["gain"]["cliffs_delta"]["ci95"], res["gain"]["stratified"]["van_elteren"]["p"], res["gain"]["mixed_model"].get("coef", float("nan"))), flush=True)
    json.dump(vs_all, open(RES / "vs_metrics_summary.json", "w"), indent=2, ensure_ascii=False)
    fam1 = [s for s in ("dude_full", "litpcba", "zinc_unmatched") if s in results]
    ps = [(s, t, results[s]["gain"]["mann_whitney"]["p"] if t == "mann_whitney" else results[s]["gain"]["stratified"]["van_elteren"]["p"]) for s in fam1 for t in ("mann_whitney", "van_elteren")]
    hp = holm([x[2] for x in ps]) if ps else []
    summ = dict(alpha=T["alpha"], family_main=[dict(set=s, test=t, p=p, holm_p=float(h)) for (s, t, p), h in zip(ps, hp)], family_stress={})
    if "zinc_charge_mismatch" in results:
        r = results["zinc_charge_mismatch"]["gain"]
        p2 = [r["mann_whitney"]["p"], r["stratified"]["van_elteren"]["p"]]; h2 = holm(p2)
        summ["family_stress"] = [dict(set="zinc_charge_mismatch", test=t, p=p, holm_p=float(h)) for t, p, h in zip(("mann_whitney", "van_elteren"), p2, h2)]
    summ["verdicts"] = {}
    for s, r in results.items():
        g = r["gain"]
        summ["verdicts"][s] = dict(mw_reject=g["mann_whitney"]["p"] < T["alpha"], stratified_reject=g["stratified"]["van_elteren"]["p"] < T["alpha"], mixed_reject=bool(g["mixed_model"].get("p", 1) < T["alpha"]),
                                   equivalent_by_cliffs_delta=g["cliffs_delta"]["tost"]["equivalent"], equivalent_by_hl=g["hodges_lehmann"]["tost"]["equivalent"], stratified_equivalent=g["stratified"]["tost"]["equivalent"],
                                   sensitivity_incl_single=dict(n=r["n_in_analysis_incl_single"], mw_p=r["gain_incl_single"]["mann_whitney"]["p"], van_elteren_p=r["gain_incl_single"]["stratified"]["van_elteren"]["p"],
                                                                cliffs_delta=r["gain_incl_single"]["cliffs_delta"]["estimate"]))
    summ["provenance"] = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RES / ("%s_per_ligand.csv" % s) for s in present], seed=T["seed_boot"], n_boot=T["n_boot_delta"], n_perm=T["n_perm"])
    summ["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(summ, open(RES / "gain_tests_summary.json", "w"), indent=2, ensure_ascii=False)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
