import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats
import decoy_stats as S

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P
import yaml

T = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))["tests"]
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RES = OUT_ROOT / "results/decoy_matching"; RES.mkdir(parents=True, exist_ok=True)
src = ROOT / "results/benchmark_analyses/swing_signal/fig4_swing_per_ligand.csv"
d = pd.read_csv(src)
d = d[d["best_default_gain"].notna()].rename(columns={"best_default_gain": "gain"}).copy()
d["is_active"] = d["is_active"].astype(int)
a, b = d[d.is_active == 1]["gain"].values, d[d.is_active == 0]["gain"].values
mw = stats.mannwhitneyu(a, b, alternative="two-sided")
rng = np.random.default_rng(T["seed_boot"] + 7000)
bd, bh = S.bootstrap_effects(d, "gain", T["n_boot_delta"], rng, n_hl=T["n_boot_hl"])
out = dict(source=str(src), n_active=len(a), n_negative=len(b), median_active=float(np.median(a)), median_negative=float(np.median(b)), mann_whitney_p=float(mw.pvalue),
           cliffs_delta=dict(estimate=float(S.cliffs_delta(a, b)), ci95=[float(x) for x in np.percentile(bd, [2.5, 97.5])], tost=S.equivalence(bd, T["cliffs_delta_equiv_bound"], T["alpha"])),
           hodges_lehmann=dict(estimate=float(S.hl_shift(a, b)), ci95=[float(x) for x in np.percentile(bh, [2.5, 97.5])], tost=S.equivalence(bh, T["hl_equiv_bound_kcal"], T["alpha"])),
           bounds=dict(cliffs_delta=T["cliffs_delta_equiv_bound"], hl_kcal=T["hl_equiv_bound_kcal"]),
           note="original gain = best empirical Vina over docked microstates minus the default microstate's best (the default was tried with more conformers in the original design); not comparable in value with the decoy-matching gain, comparable in the question asked")
out["half_width_ci90_cliffs_delta"] = float((np.percentile(bd, 95) - np.percentile(bd, 5)) / 2)
out["provenance"] = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [src, Path(S.__file__)], seed=T["seed_boot"] + 7000, n_boot=T["n_boot_delta"])
json.dump(out, open(RES / "ten_target_reanalysis.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("provenance", "source", "note")}, indent=1))
