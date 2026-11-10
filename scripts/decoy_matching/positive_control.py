import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gain_tests as TS

ROOT = TS.ROOT
P = TS.P
CFG, T, RES = TS.CFG, TS.T, TS.RES
PC = CFG["positive_control"]
df = TS.load(PC["dataset"])
use = df[df["in_analysis"] == True].copy()
rows, full = [], {}
t0 = time.time()
for k, delta in enumerate([0.0] + list(PC["deltas_kcal"])):
    d = use.copy()
    act = d["is_active"] == 1
    nd = d["score_min_nd"].astype(float)
    d["gain_inj"] = np.where(act, np.maximum(0.0, d["score_default"].astype(float) - (nd - delta)), d["gain"].astype(float))
    if delta == 0.0:
        assert np.allclose(d["gain_inj"].values, d["gain"].values.astype(float), atol=1e-6), "gain reconstruction from score_min_nd failed"
    r, _ = TS.battery(d, "gain_inj", seed=PC["seed"] % 1000 + k, n_boot=T["n_boot_delta"], n_hl=T["n_boot_hl"], n_perm=T["n_perm"])
    full[str(delta)] = r
    rows.append(dict(delta_kcal=delta, n_active=r["n_active"], n_negative=r["n_negative"], median_gain_active=r["median_active"], median_gain_negative=r["median_negative"], cliffs_delta=r["cliffs_delta"]["estimate"],
                     cliffs_delta_lo=r["cliffs_delta"]["ci95"][0], cliffs_delta_hi=r["cliffs_delta"]["ci95"][1], hl_shift=r["hodges_lehmann"]["estimate"], mw_p=r["mann_whitney"]["p"],
                     mw_reject=r["mann_whitney"]["p"] < T["alpha"], stratified_p=r["stratified"]["van_elteren"]["p"], stratified_reject=r["stratified"]["van_elteren"]["p"] < T["alpha"],
                     stratified_delta=r["stratified"]["stratified_delta"], equivalent_cliffs=r["cliffs_delta"]["tost"]["equivalent"], equivalent_hl=r["hodges_lehmann"]["tost"]["equivalent"],
                     mixed_coef=r["mixed_model"].get("coef"), mixed_p=r["mixed_model"].get("p"), mixed_reject=bool(r["mixed_model"].get("p", 1) < T["alpha"])))
    print("[%.0fs] delta %.1f: MW p=%.3g, stratified p=%.3g, equivalent(delta)=%s, mixed coef=%.3f" % (time.time() - t0, delta, rows[-1]["mw_p"], rows[-1]["stratified_p"], rows[-1]["equivalent_cliffs"], rows[-1]["mixed_coef"] or float("nan")), flush=True)
COLS = [("delta_kcal", "kcal/mol", "offset subtracted from the non-default microstate scores of the actives (0 = no injection)"), ("n_active", "", "actives in the analysis"), ("n_negative", "", "negatives in the analysis"),
        ("median_gain_active", "kcal/mol", "median injected gain, actives"), ("median_gain_negative", "kcal/mol", "median gain, negatives"), ("cliffs_delta", "", "Cliff's delta (actives vs negatives)"),
        ("cliffs_delta_lo", "", "bootstrap 2.5th percentile"), ("cliffs_delta_hi", "", "bootstrap 97.5th percentile"), ("hl_shift", "kcal/mol", "Hodges-Lehmann shift"), ("mw_p", "", "Mann-Whitney two-sided p"),
        ("mw_reject", "", "mw_p < alpha"), ("stratified_p", "", "van Elteren p"), ("stratified_reject", "", "stratified_p < alpha"), ("stratified_delta", "", "stratified Cliff's delta"),
        ("equivalent_cliffs", "", "TOST equivalence on Cliff's delta (|delta| < bound) holds"), ("equivalent_hl", "", "TOST equivalence on the HL shift holds"), ("mixed_coef", "kcal/mol", "mixed-model coefficient of is_active"),
        ("mixed_p", "", "its p"), ("mixed_reject", "", "mixed_p < alpha")]
out = RES / "power_check_injection.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow([c[0] for c in COLS])
    for r in rows:
        w.writerow(["" if r[c[0]] is None else (format(r[c[0]], ".8g") if isinstance(r[c[0]], float) else r[c[0]]) for c in COLS])
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RES / ("%s_per_ligand.csv" % PC["dataset"])], seed=PC["seed"], n_boot=T["n_boot_delta"], n_perm=T["n_perm"], extra=dict(deltas=PC["deltas_kcal"]))
P.write_schema(out, COLS, "one row per injected offset (0 + 3 deltas)", "Positive control: detection of an artificial activity-dependent gain by the decoy-matching test battery", extra=dict(provenance=BLOCK))
inj = [r for r in rows if r["delta_kcal"] > 0]
rule = all(r["stratified_reject"] for r in inj) or all(r["stratified_reject"] for r in inj if r["delta_kcal"] >= 0.6)
json.dump(dict(provenance=BLOCK, rule=PC["rule"], passed_all_deltas=all(r["stratified_reject"] for r in inj), passed_06_and_10=all(r["stratified_reject"] for r in inj if r["delta_kcal"] >= 0.6), passed=bool(rule),
               mw_reject_by_delta={str(r["delta_kcal"]): r["mw_reject"] for r in rows}, stratified_reject_by_delta={str(r["delta_kcal"]): r["stratified_reject"] for r in rows}, details=full),
          open(RES / "power_check_injection.json", "w"), indent=2, ensure_ascii=False)
print("positive control passed:", bool(rule))
sys.exit(0 if rule else 1)
