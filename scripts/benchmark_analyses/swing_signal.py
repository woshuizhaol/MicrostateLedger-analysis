import csv, json, os
import numpy as np
from scipy.stats import mannwhitneyu

V = os.environ.get("DOCKING_ROOT", ".")
D = f"{V}/verification/C2_dude_vs/outputs/figure_data/data"
OUT = "$ANALYSIS_ROOT/results/benchmark_analyses/swing_signal"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260702)

vina_by_mid, charge_by_mid, target_by_mid = {}, {}, {}
for r in csv.DictReader(open(f"{D}/fig3_prior_empirical_scatter.csv")):
    vina_by_mid[r["microstate_id"]] = float(r["empirical_best_vina"])
    charge_by_mid[r["microstate_id"]] = int(r["formal_charge"])
    target_by_mid[r["microstate_id"]] = r["target"]
print(f"[data] scatter microstates with empirical vina: {len(vina_by_mid)}")

per = []
n_default_missing = 0
n_default_no_vina = 0
n_default_not_min_charge = 0
for r in csv.DictReader(open(f"{D}/fig3_run019_rows.csv")):
    try:
        alloc = json.loads(r.get("strategy_allocation_json") or "{}")
    except json.JSONDecodeError:
        continue
    mids = set()
    for _, info in alloc.items():
        if isinstance(info, dict):
            mids |= set(info.get("candidate_mids", []))
    hit = sorted(m for m in mids if m in vina_by_mid)
    if len(hit) < 2:
        continue
    vinas = np.array([vina_by_mid[m] for m in hit])
    tgt = target_by_mid[hit[0]]
    is_active = int(r["label"])
    default_mid = None
    single = alloc.get("single")
    if isinstance(single, dict) and single.get("candidate_mids"):
        default_mid = single["candidate_mids"][0]
    bi = int(np.argmin(vinas))
    best_mid = hit[bi]
    vina_best = float(vinas[bi])
    vina_worst = float(vinas.max())
    swing = vina_worst - vina_best
    std = float(vinas.std(ddof=0))
    if default_mid is None:
        n_default_missing += 1
        vina_default = float("nan")
    elif default_mid not in vina_by_mid:
        n_default_no_vina += 1
        vina_default = float("nan")
    else:
        vina_default = float(vina_by_mid[default_mid])
        min_q_mid = min(hit, key=lambda m: (abs(charge_by_mid[m]), m))
        if min_q_mid != default_mid:
            n_default_not_min_charge += 1
    best_ne_default = None if default_mid is None else int(best_mid != default_mid)
    gain = float(vina_default - vina_best) if vina_default == vina_default else float("nan")
    per.append(dict(
        ligand_id=r.get("cid"), target=tgt, is_active=is_active,
        n_microstates=len(hit), vina_min=vina_best, vina_max=vina_worst,
        swing=round(swing, 4), std=round(std, 4),
        vina_default=(round(vina_default, 4) if vina_default == vina_default else ""),
        vina_best=round(vina_best, 4),
        default_mid=(default_mid or ""), best_mid=best_mid,
        best_ne_default=("" if best_ne_default is None else best_ne_default),
        best_default_gain=(round(gain, 4) if gain == gain else "")))

n = len(per)
print(f"[data] rankable ligands (>=2 docked microstates): {n}")
print(f"[data]   with default docked (vina available): {sum(1 for p in per if p['vina_default']!='')}")
print(f"[data]   default missing / default-no-vina / default-not-minq-within-docked: "
      f"{n_default_missing} / {n_default_no_vina} / {n_default_not_min_charge}")

def arr(key, subset=None):
    src = per if subset is None else subset
    out = []
    for p in src:
        v = p[key]
        if v == "" or v is None:
            continue
        out.append(float(v))
    return np.array(out, float)

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return (gt - lt) / (len(a) * len(b))

def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else float("nan")

def boot_median_ci(x, nb=10000):
    x = np.asarray(x, float)
    ms = np.array([np.median(x[RNG.integers(0, len(x), len(x))]) for _ in range(nb)])
    return float(np.median(x)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))

sw = arr("swing")
sw_med, sw_lo, sw_hi = boot_median_ci(sw)
swing_stats = dict(
    n_ligands=int(len(sw)),
    median=round(sw_med, 4), median_ci95=[round(sw_lo, 4), round(sw_hi, 4)],
    mean=round(float(sw.mean()), 4), std=round(float(sw.std(ddof=1)), 4),
    q1=round(float(np.percentile(sw, 25)), 4), q3=round(float(np.percentile(sw, 75)), 4),
    iqr=round(float(np.percentile(sw, 75) - np.percentile(sw, 25)), 4),
    min=round(float(sw.min()), 4), max=round(float(sw.max()), 4))
stds = arr("std")
std_stats = dict(median=round(float(np.median(stds)), 4), mean=round(float(stds.mean()), 4),
                 q1=round(float(np.percentile(stds, 25)), 4), q3=round(float(np.percentile(stds, 75)), 4))

def signal_block(key):
    a = arr(key, [p for p in per if p["is_active"] == 1])
    d = arr(key, [p for p in per if p["is_active"] == 0])
    med_a, med_d = float(np.median(a)), float(np.median(d))
    mean_a, mean_d = float(a.mean()), float(d.mean())
    U, pmw = mannwhitneyu(a, d, alternative="two-sided")
    return dict(
        n_active=int(len(a)), n_decoy=int(len(d)),
        median_active=round(med_a, 4), median_decoy=round(med_d, 4),
        abs_median_diff=round(abs(med_a - med_d), 4),
        mean_active=round(mean_a, 4), mean_decoy=round(mean_d, 4),
        abs_mean_diff=round(abs(mean_a - mean_d), 4),
        cliffs_delta=round(cliffs_delta(a, d), 4), cohens_d=round(cohens_d(a, d), 4),
        mannwhitney_U=float(U), mannwhitney_p=float(pmw))

signal_best = signal_block("vina_best")
signal_default = signal_block("vina_default")

per_tgt_signal = {}
for t in sorted({p["target"] for p in per}):
    sub = [p for p in per if p["target"] == t]
    a = arr("vina_best", [p for p in sub if p["is_active"] == 1])
    d = arr("vina_best", [p for p in sub if p["is_active"] == 0])
    if len(a) and len(d):
        per_tgt_signal[t] = round(abs(float(np.median(a)) - float(np.median(d))), 4)
signal_best_per_target_mean = round(float(np.mean(list(per_tgt_signal.values()))), 4)

cmp_ok = [p for p in per if p["best_ne_default"] != ""]
ne = sum(int(p["best_ne_default"]) for p in cmp_ok)
best_ne_default = dict(
    n_comparable=len(cmp_ok), n_best_ne_default=ne,
    frac_best_ne_default=round(ne / len(cmp_ok), 4) if cmp_ok else None,
    frac_default_is_best=round(1 - ne / len(cmp_ok), 4) if cmp_ok else None)
for lbl, name in [(1, "active"), (0, "decoy")]:
    sub = [p for p in cmp_ok if p["is_active"] == lbl]
    ne_s = sum(int(p["best_ne_default"]) for p in sub)
    best_ne_default[f"frac_{name}"] = round(ne_s / len(sub), 4) if sub else None

sw_a = arr("swing", [p for p in per if p["is_active"] == 1])
sw_d = arr("swing", [p for p in per if p["is_active"] == 0])
U_sw, p_sw = mannwhitneyu(sw_a, sw_d, alternative="two-sided")
gn_a = arr("best_default_gain", [p for p in per if p["is_active"] == 1])
gn_d = arr("best_default_gain", [p for p in per if p["is_active"] == 0])
U_gn, p_gn = mannwhitneyu(gn_a, gn_d, alternative="two-sided")
label_indep = dict(
    swing_active_median=round(float(np.median(sw_a)), 4), swing_decoy_median=round(float(np.median(sw_d)), 4),
    swing_mannwhitney_U=float(U_sw), swing_mannwhitney_p=float(p_sw),
    gain_active_median=round(float(np.median(gn_a)), 4), gain_decoy_median=round(float(np.median(gn_d)), 4),
    gain_mannwhitney_U=float(U_gn), gain_mannwhitney_p=float(p_gn),
    gain_overall_median=round(float(np.median(arr("best_default_gain"))), 4),
    gain_overall_mean=round(float(arr("best_default_gain").mean()), 4))

best_ne_default_per_target = {}
for t in sorted({p["target"] for p in per}):
    sub = [p for p in per if p["target"] == t and p["best_ne_default"] != ""]
    ne_t = sum(int(p["best_ne_default"]) for p in sub)
    best_ne_default_per_target[t] = dict(n=len(sub), frac=round(ne_t / len(sub), 4) if sub else None)

def _rep_all(alloc, kind):
    m = set()
    for _, info in alloc.items():
        if isinstance(info, dict):
            m |= set(info.get("candidate_mids", []))
    hh = sorted(x for x in m if x in vina_by_mid)
    if not hh:
        return None, 0
    vv = np.array([vina_by_mid[x] for x in hh])
    sg = alloc.get("single", {})
    dm = sg.get("candidate_mids", [None])[0] if isinstance(sg, dict) else None
    dv = vina_by_mid.get(dm) if dm else None
    val = {"best": float(vv.min()), "default": (float(dv) if dv is not None else None),
           "mean": float(vv.mean()), "worst": float(vv.max())}[kind]
    return val, len(hh)

signal_variants = {}
_rows2 = list(csv.DictReader(open(f"{D}/fig3_run019_rows.csv")))
for kind in ["best", "default", "mean", "worst"]:
    for minhit, tag in [(1, "all_docked"), (2, "ge2_docked")]:
        A2, D2 = [], []
        for r in _rows2:
            try:
                al = json.loads(r["strategy_allocation_json"] or "{}")
            except json.JSONDecodeError:
                continue
            s, nh = _rep_all(al, kind)
            if s is None or nh < minhit:
                continue
            (A2 if int(r["label"]) == 1 else D2).append(s)
        A2, D2 = np.array(A2), np.array(D2)
        signal_variants[f"{kind}__{tag}"] = dict(
            nA=int(len(A2)), nD=int(len(D2)),
            abs_median_diff=round(abs(np.median(A2) - np.median(D2)), 4),
            abs_mean_diff=round(abs(A2.mean() - D2.mean()), 4))

summary = dict(
    provenance=dict(
        scatter=f"{D}/fig3_prior_empirical_scatter.csv",
        rows=f"{D}/fig3_run019_rows.csv",
        run="ten-target docking (single seed) single-seed 20260210", seed_note="empirical_obs_count all=1 (single seed)",
        default_rule="ORDER BY ABS(formal_charge), microstate_id (single strategy candidate_mids[0])",
        best_rule="argmin empirical_best_vina over docked microstates",
        rankable_def=">=2 docked microstates with empirical vina"),
    n_scatter_microstates=len(vina_by_mid), n_rankable_ligands=n,
    swing_kcal=swing_stats, microstate_vina_std=std_stats,
    discrimination_signal_best=signal_best,
    discrimination_signal_default=signal_default,
    discrimination_signal_best_per_target=per_tgt_signal,
    discrimination_signal_best_per_target_mean=signal_best_per_target_mean,
    discrimination_signal_variants=signal_variants,
    best_ne_default=best_ne_default,
    best_ne_default_per_target=best_ne_default_per_target,
    label_independence=label_indep,
    swing_vs_signal_ratio_best=round(swing_stats["median"] / signal_best["abs_median_diff"], 3)
        if signal_best["abs_median_diff"] else None,
    original_diagnostic_reference=dict(swing="~0.85", signal="~0.51",
                                  best_ne_default="83-94%", label_indep_p="~0.88"))
json.dump(summary, open(f"{OUT}/fig4_swing_summary.json", "w"), indent=2)

cols = ["ligand_id", "target", "is_active", "n_microstates", "vina_min", "vina_max",
        "swing", "std", "vina_default", "vina_best", "best_default_gain",
        "default_mid", "best_mid", "best_ne_default"]
with open(f"{OUT}/fig4_swing_per_ligand.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for p in per:
        w.writerow({k: p[k] for k in cols})

print("\n================ Fig4a recompute (run019, single seed) ================")
print(f"rankable ligands (>=2 docked microstates): {n}")
print(f"\n[1] WITHIN-LIGAND SWING (max-min vina, kcal/mol):")
print(f"    median={swing_stats['median']}  CI95={swing_stats['median_ci95']}  "
      f"mean={swing_stats['mean']}  IQR=[{swing_stats['q1']},{swing_stats['q3']}]  "
      f"range=[{swing_stats['min']},{swing_stats['max']}]")
print(f"    (per-microstate vina std: median={std_stats['median']} mean={std_stats['mean']})")
print(f"\n[2] DISCRIMINATION SIGNAL |median(active)-median(decoy)| (kcal/mol):")
print(f"    using BEST vina    : {signal_best['abs_median_diff']}  "
      f"(act={signal_best['median_active']} dec={signal_best['median_decoy']}) "
      f"CliffDelta={signal_best['cliffs_delta']} Cohend={signal_best['cohens_d']} MWp={signal_best['mannwhitney_p']:.3g}")
print(f"    using DEFAULT vina : {signal_default['abs_median_diff']}  "
      f"(act={signal_default['median_active']} dec={signal_default['median_decoy']}) "
      f"CliffDelta={signal_default['cliffs_delta']} Cohend={signal_default['cohens_d']} MWp={signal_default['mannwhitney_p']:.3g}")
print(f"    per-target(best) mean signal: {signal_best_per_target_mean}  detail={per_tgt_signal}")
print(f"\n    SWING/SIGNAL ratio (best): {summary['swing_vs_signal_ratio_best']}")
print(f"\n[3] BEST microstate != DEFAULT(single) microstate:")
print(f"    {best_ne_default['n_best_ne_default']}/{best_ne_default['n_comparable']} = "
      f"{best_ne_default['frac_best_ne_default']}  (active={best_ne_default['frac_active']} "
      f"decoy={best_ne_default['frac_decoy']})")
print(f"\n[4] LABEL INDEPENDENCE (active vs decoy):")
print(f"    swing  MW p={label_indep['swing_mannwhitney_p']:.4f}  "
      f"(act med={label_indep['swing_active_median']} dec med={label_indep['swing_decoy_median']})")
print(f"    gain   MW p={label_indep['gain_mannwhitney_p']:.4f}  "
      f"(act med={label_indep['gain_active_median']} dec med={label_indep['gain_decoy_median']})")
print(f"    overall best-default gain: median={label_indep['gain_overall_median']} "
      f"mean={label_indep['gain_overall_mean']}")
print(f"\nsaved: {OUT}/fig4_swing_summary.json")
print(f"saved: {OUT}/fig4_swing_per_ligand.csv")
print("DONE_FIG4_SWING")
