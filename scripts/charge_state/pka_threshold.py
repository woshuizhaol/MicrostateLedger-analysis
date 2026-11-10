import csv, json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C

C.check_pins()
O = C.O
rel = [r for r in csv.DictReader(open(C.PATHS["iupac_csv"], newline="", encoding="utf-8")) if r["assessment"] == "Reliable"]
by = {}
for r in rel:
    try:
        by.setdefault(r["SMILES"], []).append((float(r["pka_value"]), r["pka_type"]))
    except ValueError:
        pass
mono = [("iupac_%d" % i, s, v[0][0]) for i, (s, v) in enumerate(by.items()) if len(v) == 1]
rows, missed = [], 0
for mid, smi, exp in mono:
    try:
        preds = [float(pk) for _, pk in O.detect_site_pkas(O.neutralize(O.canonical(smi)), C.TABLE)]
    except Exception:
        continue
    if not preds:
        missed += 1; continue
    best = min(preds, key=lambda p: abs(p - exp))
    rows.append((mid, smi, exp, best, best - exp))
err = np.array([r[4] for r in rows])
original = json.load(open(C.RM / "results/stats/pka/pka_summary.json"))["iupac"]
near = [dict(mol_id=r[0], smiles=r[1], exp=r[2], pred_full=r[3], err_full=r[4]) for r in rows if abs(abs(r[4]) - 1.0) < 0.01]
res = dict(n_records=len(mono), n_predicted=len(rows), n_silent_miss=missed,
           mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())), bias=float(err.mean()),
           k_gt1=int((np.abs(err) > 1.0).sum()), frac_gt1=float((np.abs(err) > 1.0).mean()),
           k_gt2=int((np.abs(err) > 2.0).sum()), frac_gt2=float((np.abs(err) > 2.0).mean()),
           k_flip=int(sum((r[3] > 7.4) != (r[2] > 7.4) for r in rows)),
           wilson_gt1=list(C.wilson(int((np.abs(err) > 1.0).sum()), len(rows))),
           within_0p01_of_threshold=near,
           threshold_sweep={str(t): int((np.abs(err) > t).sum()) for t in (0.95, 0.99, 1.0, 1.01, 1.05)},
           original_summary=original, matches_old_summary=dict(
               n_predicted=len(rows) == original["n_predicted"], n_silent_miss=missed == original["n_silent_miss"], mae=round(float(np.abs(err).mean()), 3) == original["pka_MAE"],
               rmse=round(float(np.sqrt((err ** 2).mean())), 3) == original["pka_RMSE"], frac_gt1=round(float((np.abs(err) > 1.0).mean()), 3) == original["frac_abs_err_gt_1"],
               frac_gt2=round(float((np.abs(err) > 2.0).mean()), 3) == original["frac_abs_err_gt_2"], flip=round(sum((r[3] > 7.4) != (r[2] > 7.4) for r in rows) / len(rows), 3) == original["pH7.4_dominant_flip_frac"]),
           provenance=C.P.provenance_block(Path(__file__), C.ROOT / "configs/charge_state.yaml", [C.PATHS["iupac_csv"], C.PATHS["dimorphite_smarts"], C.RM / "results/stats/pka/pka_summary.json"],
                                           extra=dict(original_script="$BASE_ROOT/code/analysis/pka_benchmark.py md5 2b56f7252c7108d36da99b86243d52c0", rounding="none")))
(C.RES).mkdir(parents=True, exist_ok=True)
json.dump(res, open(C.RES / "pka_threshold_sensitivity.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("provenance", "original_summary")}, indent=1))
sys.exit(0 if all(res["matches_old_summary"].values()) else 1)
