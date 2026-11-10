import itertools, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import casf_common as C

C.check_pins()
ST, PRED = C.CFG["stats"], C.CFG["predictors"]["list"]
import csv
rows_in = list(csv.DictReader(open(C.RAW / "per_complex_scores_by_predictor.csv", encoding="utf-8")))
rows = []
for sc in C.SCORERS:
    sel = [r for r in rows_in if r["scorer"] == sc]
    pids = sorted({r["pdbid"] for r in sel}); n = len(pids)
    y = np.array([float(next(r for r in sel if r["pdbid"] == p and r["predictor"] == PRED[0])["logKa"]) for p in pids])
    single = np.array([float(next(r for r in sel if r["pdbid"] == p and r["predictor"] == PRED[0])["score_single"]) for p in pids])
    S = {pr: np.array([float(next(r for r in sel if r["pdbid"] == p and r["predictor"] == pr)["score_pred"]) for p in pids]) for pr in PRED}
    idx = C.boot_indices(n, ST["n_boot"], ST["seed_boot"])
    Rp_s = C.rp_rmse(y, single)[0]
    for a, b in itertools.combinations(PRED, 2):
        Ra, Rb = C.rp_rmse(y, S[a])[0], C.rp_rmse(y, S[b])[0]
        d, _ = C.paired_delta_boot(y, S[b], S[a], idx)
        lo, hi = C.pct(d)
        rows.append(dict(scorer=sc, predictor_a=a, predictor_b=b, n=n, dRp_a=Ra - Rp_s, dRp_b=Rb - Rp_s, diff=Ra - Rb, lo=lo, hi=hi,
                         p_two_sided=min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))), n_states_differ=int(sum(1 for x, z in zip(S[a], S[b]) if abs(x - z) > C.CFG["affected_subset"]["score_change_tol"]))))
COLS = [("scorer", "", "vina, smina, ad4"), ("predictor_a", "", "first predictor"), ("predictor_b", "", "second predictor"), ("n", "", "n_common complexes"),
        ("dRp_a", "", "Rp(state of A) - Rp(default)"), ("dRp_b", "", "Rp(state of B) - Rp(default)"), ("diff", "", "Rp(A) - Rp(B)"),
        ("lo", "", "paired bootstrap 2.5th percentile of diff (same resamples as predictor_robustness.csv)"), ("hi", "", "paired bootstrap 97.5th percentile"),
        ("p_two_sided", "", "min(1, 2 * min(P(diff <= 0), P(diff >= 0))) over the bootstrap resamples"), ("n_states_differ", "", "complexes in which the two predictors give a different score")]
BLOCK = C.P.provenance_block(Path(__file__), C.ROOT / "configs/casf_scoring.yaml", [C.RAW / "per_complex_scores_by_predictor.csv"], seed=None, n_boot=ST["n_boot"], extra=dict(seed_boot=ST["seed_boot"], post_hoc=True))
out = C.RES / "predictor_contrast.csv"
C.write_csv(out, COLS, rows)
C.P.write_schema(out, COLS, "scorer x predictor pair", "POST HOC paired bootstrap contrasts of Rp between predictor-defined dominant states", extra=dict(provenance=BLOCK))
for r in rows:
    if r["predictor_a"] == "dimorphite_dom":
        C.log("%-5s dimorphite_dom - %-13s diff %+.4f [%+.4f, %+.4f] p=%.3f" % (r["scorer"], r["predictor_b"], r["diff"], r["lo"], r["hi"], r["p_two_sided"]))
