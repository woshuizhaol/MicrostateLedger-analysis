import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P
import decoy_stats as S

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
T = CFG["tests"]
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RES = OUT_ROOT / "results/decoy_matching"
SETS = ["dude_full", "litpcba", "zinc_unmatched", "zinc_charge_mismatch"]
NB, NP, SEED = 2000, 5000, 20261101
VARIANTS = ["primary", "charge_sites_ndocked", "ndocked_heavy", "target_ndocked", "target_ndocked_heavy"]
t0 = time.time()


def load(set_name):
    df = pd.read_csv(RES / ("%s_per_ligand.csv" % set_name), dtype={"ligand_id": str, "charge_class": str, "sites_class": str}, keep_default_na=False, na_values=[""])
    df["is_active"] = (df["role"] == "active").astype(int)
    for c in ("in_analysis", "in_prefix", "docked_record"):
        df[c] = df[c].astype(str) == "True"
    df["default_ok"] = df["default_docked"].astype(str) == "True"
    return df


def bins(x, edges, labels):
    return pd.cut(pd.Series(x).astype(float), edges, labels=labels).astype(str)


def strata_for(d, variant):
    nd = d["n_docked"].astype(int)
    if variant == "primary":
        s, _ = S.build_strata(d)
        return s
    dfine = "D" + np.minimum(nd, 14).astype(str).str.zfill(2)
    dcoarse = bins(nd, [1, 3, 6, 10, 10 ** 6], ["D2-3", "D4-6", "D7-10", "D11+"])
    hfine = bins(d["n_heavy"], [0, 20, 25, 30, 35, 10 ** 6], ["H<=20", "H21-25", "H26-30", "H31-35", "H>35"])
    hcoarse = bins(d["n_heavy"], [0, 25, 32, 10 ** 6], ["H<=25", "H26-32", "H>32"])
    if variant == "charge_sites_ndocked":
        s = d["charge_class"].astype(str) + "|" + d["sites_class"].astype(str) + "|" + dfine
        s[d["charge_class"].isna() | d["sites_class"].isna()] = np.nan
    elif variant == "ndocked_heavy":
        s = ("D" + np.minimum(nd, 12).astype(str).str.zfill(2)) + "|" + hfine
        s[d["n_heavy"].isna()] = np.nan
    elif variant == "target_ndocked":
        s = d["target"].astype(str) + "|" + dcoarse
    elif variant == "target_ndocked_heavy":
        s = d["target"].astype(str) + "|" + dcoarse + "|" + hcoarse
        s[d["n_heavy"].isna()] = np.nan
    else:
        raise ValueError(variant)
    return s


def one(d, variant, k):
    s = strata_for(d, variant)
    valid = ~s.isna().values
    sub = d[valid].reset_index(drop=True); ss = s[valid].reset_index(drop=True)
    g = sub["gain"].values.astype(float); act = sub["is_active"].values.astype(bool)
    ve, tab = S.van_elteren(g, act, ss.values)
    sd = S.stratified_delta(tab)
    codes = pd.factorize(ss)[0]; ncodes = int(codes.max()) + 1
    cells = S.cell_index(sub)
    rng = np.random.default_rng(SEED + k)
    bs = np.empty(NB)
    for i in range(NB):
        ix = S.boot_draws(cells, rng)
        bs[i] = S.strat_delta_fast(g[ix], act[ix], codes[ix], ncodes)
    bs = bs[np.isfinite(bs)]
    perm_p, _ = S.permutation_pvalue(g, act, ss.values, NP, np.random.default_rng(SEED + 1000 + k))
    eq = S.equivalence(bs, T["cliffs_delta_equiv_bound"], T["alpha"])
    return dict(n_ligands_in_analysis=int(len(d)), n_ligands_with_stratum=int(valid.sum()), n_active_used=int(act.sum()), n_negative_used=int((~act).sum()), n_strata_total=int(tab.shape[0]), n_strata_used=int(tab["used"].sum()),
                n_ligands_in_used_strata=int(ve["n_used"]), van_elteren_z=float(ve["z"]), van_elteren_p=float(ve["p"]), stratified_delta=float(sd), delta_ci95_lo=float(np.percentile(bs, 2.5)), delta_ci95_hi=float(np.percentile(bs, 97.5)),
                tost_ci90_lo=float(eq["ci90"][0]), tost_ci90_hi=float(eq["ci90"][1]), tost_p=float(eq["p_tost"]), equivalent=bool(eq["equivalent"]), permutation_p=float(perm_p)), ve


rows, checks = [], []
for si, name in enumerate(SETS):
    df = load(name)
    d = df[df["in_analysis"]].dropna(subset=["gain", "n_docked"]).reset_index(drop=True)
    for vi, variant in enumerate(VARIANTS):
        r, ve = one(d, variant, 10 * si + vi)
        r.update(set=name, post_hoc_stratification=variant)
        rows.append(r)
        print("[%5.0fs] %-22s %-22s used %5d/%5d ligands in %3d strata | VE z %6.2f p %.3g | strat. delta %6.3f [%6.3f, %6.3f] | TOST equivalent %-5s | perm p %.3g" % (
            time.time() - t0, name, variant, r["n_ligands_in_used_strata"], r["n_ligands_in_analysis"], r["n_strata_used"], r["van_elteren_z"], r["van_elteren_p"], r["stratified_delta"], r["delta_ci95_lo"], r["delta_ci95_hi"], r["equivalent"], r["permutation_p"]), flush=True)
        if variant == "primary":
            jf = RES / ("gain_tests_%s.json" % name)
            if jf.exists():
                ref = json.load(open(jf))["gain"]["stratified"]["van_elteren"]
                ok = abs(ref["z"] - ve["z"]) < 1e-9 * max(1.0, abs(ref["z"])) and abs(ref["p"] - ve["p"]) < 1e-9 * max(1e-300, abs(ref["p"])) + 1e-15 and ref["n_used"] == ve["n_used"]
                checks.append((name, ok, ref["z"], ve["z"], ref["p"], ve["p"]))
                print("         consistency with the primary stratified test of %s: %s (stored z %.6f p %.6g | here z %.6f p %.6g)" % (name, "OK" if ok else "MISMATCH", ref["z"], ref["p"], ve["z"], ve["p"]), flush=True)
res = pd.DataFrame(rows)
cols = ["set", "post_hoc_stratification", "n_ligands_in_analysis", "n_ligands_with_stratum", "n_active_used", "n_negative_used", "n_strata_total", "n_strata_used", "n_ligands_in_used_strata", "van_elteren_z", "van_elteren_p", "stratified_delta",
        "delta_ci95_lo", "delta_ci95_hi", "tost_ci90_lo", "tost_ci90_hi", "tost_p", "equivalent", "permutation_p"]
out1 = RES / "posthoc_matched_tests.csv"
res[cols].to_csv(out1, index=False, float_format="%.8g")

ex = []
for name in SETS:
    df = load(name); p = df[df["in_prefix"]]
    for (tg, role), g in p.groupby(["target", "role"]):
        no_rec = int((~g["docked_record"]).sum())
        dfail = int((g["docked_record"] & ~g["default_ok"]).sum())
        single = int((g["docked_record"] & g["default_ok"] & (g["n_docked"].fillna(0) < 2)).sum())
        used = int(g["in_analysis"].sum())
        ex.append(dict(set=name, target=tg, role=role, n_in_prefix=int(len(g)), n_in_analysis=used, n_no_docking_record=no_rec, n_default_state_failed=dfail, n_single_docked_microstate=single, excluded_fraction=1 - used / len(g)))
exd = pd.DataFrame(ex)
out2 = RES / "posthoc_exclusion_by_role_target.csv"
exd.to_csv(out2, index=False, float_format="%.6g")
tot = exd.groupby(["set", "role"])[["n_in_prefix", "n_in_analysis", "n_no_docking_record", "n_default_state_failed", "n_single_docked_microstate"]].sum()
tot["excluded_fraction"] = 1 - tot["n_in_analysis"] / tot["n_in_prefix"]
print("\nExclusion of ligands of the analysis prefix by set and role:\n" + tot.round(4).to_string())

BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RES / ("%s_per_ligand.csv" % s) for s in SETS], seed=SEED, n_boot=NB, n_perm=NP, extra=dict(post_hoc=True, consistency_checks=[dict(set=c[0], ok=bool(c[1])) for c in checks]))
D1 = {"set": "ligand set", "post_hoc_stratification": "POST HOC stratification (see the script header)", "n_ligands_in_analysis": "ligands of the analysis unit (in_analysis)", "n_ligands_with_stratum": "of them with a defined stratum",
      "n_active_used": "actives with a defined stratum", "n_negative_used": "negatives with a defined stratum", "n_strata_total": "strata with at least one ligand", "n_strata_used": "strata containing both roles (the only ones that enter the test)",
      "n_ligands_in_used_strata": "ligands in those strata", "van_elteren_z": "van Elteren Z (positive = actives have the larger gain)", "van_elteren_p": "two-sided p of Z", "stratified_delta": "weighted mean within-stratum Cliff's delta (positive = actives larger)",
      "delta_ci95_lo": "ligand bootstrap (within target x role, 2,000) 2.5%", "delta_ci95_hi": "97.5%", "tost_ci90_lo": "5% of the bootstrap distribution", "tost_ci90_hi": "95%", "tost_p": "TOST p (max of the two one-sided bootstrap tail shares)",
      "equivalent": "90% interval inside (-0.147, +0.147)", "permutation_p": "within-stratum label permutation p of the van Elteren T (5,000)"}
P.write_schema(out1, [(c, "", D1[c]) for c in cols], "one row per ligand set x post hoc stratification", "POST HOC sensitivity of the stratified activity-independence test (finer control of n_docked, heavy atoms and target)", extra=dict(provenance=BLOCK))
D2 = {"set": "ligand set", "target": "receptor target", "role": "active / negative", "n_in_prefix": "ligands of the analysis prefix (slice <= max_slice, round <= max_round)", "n_in_analysis": "of them in the gain analysis",
      "n_no_docking_record": "no docking record at all", "n_default_state_failed": "docking record present, default microstate not docked", "n_single_docked_microstate": "default docked but no other microstate docked (gain not defined)",
      "excluded_fraction": "1 - n_in_analysis / n_in_prefix"}
P.write_schema(out2, [(c, "", D2[c]) for c in exd.columns], "one row per set x target x role", "Why ligands of the analysis prefix are not in the gain analysis", extra=dict(provenance=BLOCK))
bad = [c for c in checks if not c[1]]
print("\nwrote %s and %s | consistency checks: %d, mismatches: %d | %.0f s" % (out1, out2, len(checks), len(bad), time.time() - t0))
sys.exit(1 if bad else 0)
