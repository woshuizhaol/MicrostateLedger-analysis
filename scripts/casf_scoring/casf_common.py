import json, math, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
RM = Path(os.environ.get("BASE_ROOT", "."))
VF = Path(os.environ.get("DOCKING_ROOT", "."))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/utils")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts/charge_state"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/casf_scoring.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("CASF_OUT_ROOT", str(ROOT)))
RES, RAW = OUT_ROOT / "results/casf_scoring", OUT_ROOT / "raw_outputs/casf_scoring"
SCORERS = CFG["scorers"]
CONDS = ["input", "neutral", "single", "dominant", "enum_best", "enum_best_all", "random"]


def E(s):
    return Path(s.replace("$BASE_ROOT", str(RM)).replace("$DOCKING_ROOT", str(VF)).replace("$ANALYSIS_ROOT", str(ROOT)))


PATHS = {k: E(v) for k, v in CFG["paths"].items()}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def check_pins():
    bad = [k for k, m in CFG["pinned_md5"].items() if P.file_md5(PATHS[k]) != m]
    for name, d in (("scores", PATHS["d11_scores"]), ("scores_ad4", PATHS["d11_ad4"])):
        out = subprocess.run("cd %s && md5sum *.json | sort -k2 | md5sum" % d, shell=True, capture_output=True, text=True).stdout.split()[0]
        if out != CFG["d11_score_digest"][name]:
            bad.append("d11_score_digest:" + name)
    if bad:
        sys.exit("pinned input changed: %s" % bad)


fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)


def load_d11():
    import glob
    main, ad4 = {}, {}
    for f in sorted(glob.glob(str(PATHS["d11_scores"] / "*.json"))):
        j = json.load(open(f)); main[j["pdbid"]] = j
    for f in sorted(glob.glob(str(PATHS["d11_ad4"] / "*.json"))):
        j = json.load(open(f)); ad4[j["pdbid"]] = j
    aff = {}
    for ln in open(PATHS["coreset_dat"]):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split(); aff[t[0]] = float(t[3])
    assert set(main) == set(ad4) == set(aff), "D11 file sets differ"
    return main, ad4, aff


def stored_cond_score(main, ad4, pid, cond, scorer):
    m, a = main[pid], ad4[pid]
    if scorer in ("vina", "smina"):
        if cond == "enum_best":
            return m["conditions"].get("enum_best" if scorer == "vina" else "enum_best_smina", {}).get(scorer)
        return m["conditions"].get(cond, {}).get(scorer)
    if cond == "enum_best":
        return a["conditions"].get("enum_best_ad4", {}).get("ad4")
    if cond == "enum_best_all":
        return a["conditions"].get("enum_best_all_ad4", {}).get("ad4")
    return a["conditions"].get(cond, {}).get("ad4")


def member_score(main, ad4, pid, smiles, scorer):
    if scorer == "ad4":
        return ad4[pid]["all_ad4"].get(smiles)
    return main[pid]["all_scored"].get(smiles, {}).get(scorer)


def rp_rmse_rows(y, X):
    y = np.asarray(y, float)
    sx = X - X.mean(1, keepdims=True)
    sy = y - y.mean()
    sxx, syy = (sx * sx).sum(1), float((sy * sy).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -(sx @ sy) / np.sqrt(sxx * syy)
        rmse = np.sqrt(syy / len(y) * (1.0 - r * r))
    return r, rmse


def rp_rmse(y, x):
    r, m = rp_rmse_rows(y, np.asarray(x, float)[None, :])
    return float(r[0]), float(m[0])


def boot_indices(n, B, seed):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def paired_delta_boot(y, a, b, idx):
    Y = y[idx]
    A, Bm = a[idx], b[idx]
    def rows(Xm):
        sx = Xm - Xm.mean(1, keepdims=True); sy = Y - Y.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -(sx * sy).sum(1) / np.sqrt((sx * sx).sum(1) * (sy * sy).sum(1))
            rmse = np.sqrt((sy * sy).sum(1) / Y.shape[1] * (1.0 - r * r))
        return r, rmse
    ra, ma = rows(A); rb, mb = rows(Bm)
    return rb - ra, mb - ma


def pct(x, lo=None, hi=None):
    lo = 100 * CFG["stats"]["alpha"] / 2 if lo is None else lo
    hi = 100 * (1 - CFG["stats"]["alpha"] / 2) if hi is None else hi
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (float("nan"), float("nan"))


def fmt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return format(float(v), ".10g")
    return str(v)


def write_csv(path, cols, rows):
    import csv
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = [c[0] for c in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(names)
        for r in rows:
            extra = set(r) - set(names)
            if extra:
                raise ValueError("%s: unexpected keys %s" % (path.name, sorted(extra)))
            w.writerow([fmt(r.get(n)) for n in names])


def load_tables():
    import csv
    main, ad4, aff = load_d11()
    PIDS = sorted(main)
    M = {p: [] for p in PIDS}
    for r in csv.DictReader(open(RAW / "per_microstate_inplace_scores.csv", encoding="utf-8")):
        M[r["pdbid"]].append(dict(smiles=r["smiles"], q=int(r["q"]), roles=set(filter(None, r["roles"].split(";"))), in_ens=r["in_ensemble"] == "True",
                                  score={sc: (float(r[sc]) if r[sc] != "" else None) for sc in SCORERS}))
    PS = {}
    for r in csv.DictReader(open(RAW / "predictor_states.csv", encoding="utf-8")):
        PS[(r["pdbid"], r["predictor"])] = r
    common = {sc: [p for p in PIDS if all(fin(stored_cond_score(main, ad4, p, c, sc)) for c in CONDS)] for sc in SCORERS}
    return PIDS, aff, M, PS, common


def uniform_matrix(n_cols, B, seed0):
    U = np.empty((B, n_cols))
    for b in range(1, B + 1):
        U[b - 1] = np.random.default_rng(seed0 + b).random(n_cols)
    return U


def null_rows(y, s_single, pools, U):
    n = len(y)
    L = np.array([len(p) for p in pools])
    Lc = np.maximum(L, 1)
    P = np.full((n, max(1, int(L.max()))), np.nan)
    for i, p in enumerate(pools):
        P[i, :len(p)] = p
    idx = np.minimum((U * Lc[None, :]).astype(int), Lc[None, :] - 1)
    X = P[np.arange(n)[None, :], idx]
    X = np.where((L == 0)[None, :], np.asarray(s_single, float)[None, :], X)
    return rp_rmse_rows(y, X)


def null_summary(d_null, obs, B_expected=None):
    d = np.asarray(d_null, float)
    q025, q975 = pct(d)
    return dict(null_mean=float(d.mean()), q025=float(q025), q975=float(q975), obs_dRp=float(obs), p_emp=float((np.sum(d >= obs) + 1) / (len(d) + 1)),
                obs_within_null_95=bool(q025 <= obs <= q975), frac_null_ge_obs=float(np.mean(d >= obs)))
