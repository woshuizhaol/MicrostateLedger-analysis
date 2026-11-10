import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RES = OUT_ROOT / "results/decoy_matching"
SETS = ["dude_full", "litpcba", "zinc_unmatched", "zinc_charge_mismatch"]
EDGES, LABELS = [0, 20, 25, 30, 35, 10 ** 6], ["<=20", "21-25", "26-30", "31-35", ">35"]

sw, hb = [], []
for s in SETS:
    df = pd.read_csv(RES / ("%s_per_ligand.csv" % s), dtype={"ligand_id": str}, keep_default_na=False, na_values=[""])
    d = df[df["in_analysis"].astype(str) == "True"].copy()
    act = (d["role"] == "active").values
    a, b = d.loc[act, "swing"].values.astype(float), d.loc[~act, "swing"].values.astype(float)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    sw.append(dict(set=s, n_active=len(a), n_negative=len(b), median_swing_active=float(np.median(a)), median_swing_negative=float(np.median(b)), mw_p=float(mw.pvalue), cliffs_delta=float(2 * mw.statistic / (len(a) * len(b)) - 1)))
    d = d.dropna(subset=["gain", "n_heavy"])
    d["bin"] = pd.cut(d["n_heavy"].astype(float), EDGES, labels=LABELS).astype(str)
    for role in ("active", "negative"):
        r = d[d["role"] == role]
        hb.append(dict(set=s, role=role, heavy_atom_bin="ALL", n=len(r), median_gain=float(r["gain"].median()), mean_heavy_atoms=float(r["n_heavy"].mean()), median_heavy_atoms=float(r["n_heavy"].median())))
        for lab in LABELS:
            q = r[r["bin"] == lab]
            hb.append(dict(set=s, role=role, heavy_atom_bin=lab, n=len(q), median_gain=float(q["gain"].median()) if len(q) else float("nan"), mean_heavy_atoms=float(q["n_heavy"].mean()) if len(q) else float("nan"),
                           median_heavy_atoms=float(q["n_heavy"].median()) if len(q) else float("nan")))
    print("%-22s swing n %d/%d | median %.3f/%.3f | MW p %.3g | Cliff delta %+.4f" % (s, len(a), len(b), np.median(a), np.median(b), mw.pvalue, sw[-1]["cliffs_delta"]), flush=True)

o1, o2 = RES / "posthoc_swing_tests.csv", RES / "posthoc_heavy_atom_bins.csv"
pd.DataFrame(sw).to_csv(o1, index=False, float_format="%.8g")
pd.DataFrame(hb).to_csv(o2, index=False, float_format="%.8g")
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RES / ("%s_per_ligand.csv" % s) for s in SETS], seed=None, extra=dict(post_hoc=True))
P.write_schema(o1, [("set", "", "ligand set"), ("n_active", "", "actives in the analysis unit (in_analysis)"), ("n_negative", "", "negatives in the analysis unit"), ("median_swing_active", "kcal/mol", "median of max - min Vina score over the docked microstates, actives"),
               ("median_swing_negative", "kcal/mol", "same, negatives"), ("mw_p", "", "two-sided Mann-Whitney p"), ("cliffs_delta", "", "2U/(n_a n_b) - 1; positive = actives have the larger swing (point estimate, no interval)")],
               "one row per ligand set", "POST HOC: swing (quantity of the original manuscript statement) tested between actives and negatives on the full data", extra=dict(provenance=BLOCK))
P.write_schema(o2, [("set", "", "ligand set"), ("role", "", "active / negative"), ("heavy_atom_bin", "", "ALL or a bin of the heavy-atom count (<=20, 21-25, 26-30, 31-35, >35)"), ("n", "", "ligands of the analysis unit in the bin"),
               ("median_gain", "kcal/mol", "median gain in the bin"), ("mean_heavy_atoms", "", "mean heavy-atom count in the bin"), ("median_heavy_atoms", "", "median heavy-atom count in the bin")],
               "one row per ligand set x role x bin (bin ALL = the whole analysis unit)", "POST HOC: gain medians of actives and negatives inside heavy-atom bins", extra=dict(provenance=BLOCK))
print("wrote %s (%d rows) and %s (%d rows)" % (o1, len(sw), o2, len(hb)))
