import json, os, sys, time, warnings
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

OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RES = OUT_ROOT / "results/decoy_matching"; RES.mkdir(parents=True, exist_ok=True)
SEED, R, N_A, N_B = 20261024, 300, 300, 900
rng = np.random.default_rng(SEED)
ps = dict(mann_whitney=[], van_elteren=[], mixed=[], mixed_rank=[])
t0 = time.time()
for r in range(R):
    n = N_A + N_B
    df = pd.DataFrame(dict(target=rng.choice(["a", "b", "c", "d"], n), is_active=np.r_[np.ones(N_A, int), np.zeros(N_B, int)], charge_class=rng.choice(["<=-1", "0", "+1", ">=+2"], n),
                           sites_class=rng.choice(["0", "1", ">=2"], n), n_eligible=rng.integers(2, 25, n), n_heavy=rng.integers(15, 40, n)))
    base = rng.gamma(1.2, 0.25, n) * (1 + 0.04 * df.n_eligible.values); base[rng.random(n) < 0.17] = 0.0
    df["gain"] = base
    g, act = df.gain.values, df.is_active.values.astype(bool)
    ps["mann_whitney"].append(stats.mannwhitneyu(g[act], g[~act], alternative="two-sided").pvalue)
    s, _ = S.build_strata(df); ps["van_elteren"].append(S.van_elteren(g, act, s.values)[0]["p"])
    ps["mixed"].append(S.mixed_model(df, "gain").get("p", np.nan)); ps["mixed_rank"].append(S.mixed_model_rank(df, "gain").get("p", np.nan))
out = {}
for k, v in ps.items():
    v = np.array(v, float); f = v[np.isfinite(v)]
    k_rej = int((f < 0.05).sum())
    lo, hi = stats.binomtest(k_rej, len(f), 0.05).proportion_ci(confidence_level=0.95, method="wilson")
    out[k] = dict(n=len(f), n_nonfinite=int(len(v) - len(f)), rejection_rate=k_rej / len(f), rejection_rate_ci95=[float(lo), float(hi)], ks_uniform_p=float(stats.kstest(f, "uniform").pvalue))
summ = dict(provenance=P.provenance_block(Path(__file__), None, [Path(S.__file__)], seed=SEED, extra=dict(replicates=R, n_active=N_A, n_negative=N_B)), by_test=out, wall_seconds=round(time.time() - t0, 1),
            note="synthetic null: gain skewed, 17% zeros, depends on n_eligible, independent of activity; a calibrated test rejects ~5%")
json.dump(summ, open(RES / "null_calibration.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}, indent=1))
