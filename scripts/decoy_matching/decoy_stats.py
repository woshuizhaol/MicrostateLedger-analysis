import math
import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.rankdata(np.concatenate([a, b]))
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return 2.0 * u / (len(a) * len(b)) - 1.0


def hl_shift(a, b, tol=1e-5):
    a, b = np.asarray(a, float), np.sort(np.asarray(b, float))
    na, nb = len(a), len(b)
    lo, hi = float(a.min() - b.max()), float(a.max() - b.min())
    target = 0.5 * na * nb
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        cnt = float((nb - np.searchsorted(b, a - mid, side="left")).sum())
        if cnt >= target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cell_index(df, cells=("target", "is_active")):
    return [np.asarray(ix) for _, ix in df.groupby(list(cells)).indices.items()]


def boot_draws(cells, rng):
    return np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def bootstrap_effects(df, gaincol, n_boot, rng, do_hl=False, n_hl=0):
    g = df[gaincol].values.astype(float); act = df["is_active"].values.astype(bool)
    cells = cell_index(df)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ix = boot_draws(cells, rng)
        d[i] = cliffs_delta(g[ix][act[ix]], g[ix][~act[ix]])
    h = np.empty(n_hl)
    for i in range(n_hl):
        ix = boot_draws(cells, rng)
        h[i] = hl_shift(g[ix][act[ix]], g[ix][~act[ix]])
    return d, h


def equivalence(boot, bound, alpha=0.05):
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    p_low, p_up = float(np.mean(boot <= -bound)), float(np.mean(boot >= bound))
    return dict(ci90=[float(lo), float(hi)], p_lower=p_low, p_upper=p_up, p_tost=max(p_low, p_up), equivalent=bool(lo > -bound and hi < bound))


def build_strata(df, extra_cuts=None):
    q = np.quantile(df["n_eligible"].astype(float), [0.25, 0.5, 0.75]) if extra_cuts is None else extra_cuts
    qi = np.digitize(df["n_eligible"].astype(float).values, q, right=True)
    s = df["charge_class"].astype(str) + "|" + df["sites_class"].astype(str) + "|Q" + pd.Series(qi + 1, index=df.index).astype(str)
    s[df["charge_class"].isna() | df["sites_class"].isna()] = np.nan
    return s, q


def van_elteren(g, act, strata):
    g = np.asarray(g, float); act = np.asarray(act, bool); strata = np.asarray(strata, object)
    T = V = 0.0; rows = []; nuse = 0; nstr = 0
    for s in pd.unique(strata[~pd.isna(strata)]):
        m = strata == s
        gs, as_ = g[m], act[m]
        a, n = int(as_.sum()), int(m.sum()); b = n - a
        if a == 0 or b == 0 or n < 2:
            rows.append(dict(stratum=s, n_active=a, n_negative=b, used=False)); continue
        r = stats.rankdata(gs)
        W = r[as_].sum(); E = a * (n + 1) / 2.0
        var = a * b / (n * (n - 1.0)) * (float((r ** 2).sum()) - n * (n + 1) ** 2 / 4.0)
        w = 1.0 / (n + 1.0)
        T += w * (W - E); V += w * w * var; nuse += n; nstr += 1
        u = W - a * (a + 1) / 2.0
        rows.append(dict(stratum=s, n_active=a, n_negative=b, used=True, median_active=float(np.median(gs[as_])), median_negative=float(np.median(gs[~as_])), cliffs_delta=2 * u / (a * b) - 1,
                         weight=a * b / (n + 1.0)))
    z = T / math.sqrt(V) if V > 0 else float("nan")
    p = 2 * stats.norm.sf(abs(z)) if V > 0 else float("nan")
    return dict(z=float(z), p=float(p), T=float(T), n_strata=nstr, n_used=nuse), pd.DataFrame(rows)


def stratified_delta(tab):
    t = tab[tab["used"]]
    return float((t["weight"] * t["cliffs_delta"]).sum() / t["weight"].sum()) if len(t) else float("nan")


def strat_delta_fast(g, act, codes, n_codes):
    num = den = 0.0
    for c in range(n_codes):
        m = codes == c
        if not m.any():
            continue
        a = int(act[m].sum()); n = int(m.sum()); b = n - a
        if a == 0 or b == 0:
            continue
        r = stats.rankdata(g[m]); u = r[act[m]].sum() - a * (a + 1) / 2.0
        w = a * b / (n + 1.0); num += w * (2 * u / (a * b) - 1); den += w
    return num / den if den else float("nan")


def permutation_pvalue(g, act, strata, n_perm, rng):
    g = np.asarray(g, float); act = np.asarray(act, bool); strata = np.asarray(strata, object)
    blocks = []
    T0 = 0.0
    for s in pd.unique(strata[~pd.isna(strata)]):
        m = strata == s; as_ = act[m]; a, n = int(as_.sum()), int(m.sum())
        if a == 0 or a == n or n < 2:
            continue
        r = stats.rankdata(g[m]); w = 1.0 / (n + 1.0)
        blocks.append((r, a, n, w)); T0 += w * (r[as_].sum() - a * (n + 1) / 2.0)
    Tn = np.zeros(n_perm)
    for r, a, n, w in blocks:
        for lo in range(0, n_perm, 2000):
            hi = min(n_perm, lo + 2000)
            idx = np.argsort(rng.random((hi - lo, n)), axis=1)[:, :a]
            Tn[lo:hi] += w * (r[idx].sum(1) - a * (n + 1) / 2.0)
    p = (np.sum(np.abs(Tn) >= abs(T0) - 1e-12) + 1.0) / (n_perm + 1.0)
    return float(p), float(T0)


def mixed_model(df, gaincol):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[gaincol, "charge_class", "sites_class", "n_eligible", "n_heavy"]).copy()
    d["y"] = d[gaincol].astype(float); d["is_active"] = d["is_active"].astype(int)
    d["n_el10"] = d["n_eligible"].astype(float) / 10.0; d["n_hv10"] = d["n_heavy"].astype(float) / 10.0
    out = dict(n=len(d), n_groups=int(d["target"].nunique()), model="MixedLM")
    f = "y ~ is_active + C(charge_class) + C(sites_class) + n_el10 + n_hv10"
    try:
        m = smf.mixedlm(f, d, groups=d["target"]).fit(reml=True, method=["lbfgs"])
        if not m.converged:
            raise RuntimeError("not converged")
        ci = m.conf_int().loc["is_active"]
        out.update(coef=float(m.params["is_active"]), se=float(m.bse["is_active"]), ci95=[float(ci[0]), float(ci[1])], p=float(m.pvalues["is_active"]), converged=True, group_var=float(m.cov_re.iloc[0, 0]))
    except Exception as exc:
        try:
            m = smf.gee(f, groups="target", data=d, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = m.conf_int().loc["is_active"]
            out.update(model="GEE (MixedLM failed: %s)" % str(exc)[:60], coef=float(m.params["is_active"]), se=float(m.bse["is_active"]), ci95=[float(ci[0]), float(ci[1])], p=float(m.pvalues["is_active"]), converged=True)
        except Exception as exc2:
            out.update(model="failed", converged=False, error=str(exc2)[:80])
    return out


def mixed_model_rank(df, gaincol):
    d = df.copy(); d[gaincol + "_rank"] = stats.rankdata(d[gaincol].astype(float)) / len(d)
    return mixed_model(d, gaincol + "_rank")


def _pairs(y, s):
    s = np.asarray(s, float); y = np.asarray(y, int)
    fin = np.isfinite(s)
    if not fin.any():
        return None
    sentinel = s[fin].max() + max(1e-3, abs(s[fin].max()) * 1e-3)
    s = np.where(fin, s, sentinel)
    return y, -s


def auc(y, s):
    p = _pairs(y, s)
    if p is None:
        return float("nan")
    y, r = p
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rk = stats.rankdata(r)
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ef(y, s, pct):
    p = _pairs(y, s)
    if p is None:
        return float("nan")
    y, r = p
    n = len(y); n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    k = max(1, int(math.ceil(n * pct / 100.0)))
    top = np.argsort(-r, kind="stable")[:k]
    return float(y[top].sum() / (k * n1 / n))


def bedroc(y, s, alpha=20.0):
    from rdkit.ML.Scoring import Scoring
    p = _pairs(y, s)
    if p is None:
        return float("nan")
    y, r = p
    order = np.argsort(-r, kind="stable")
    try:
        return float(Scoring.CalcBEDROC([[bool(v)] for v in y[order]], 0, float(alpha)))
    except Exception:
        return float("nan")
