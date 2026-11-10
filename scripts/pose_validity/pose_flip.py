import math
import numpy as np
import pandas as pd

CRITERIA = ["geom", "full_pb", "score_rank"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def score_rank_pass(scores):
    s = np.asarray(scores, float); n = len(s)
    rank = np.array([1 + int((s < x).sum()) for x in s])
    return rank <= math.ceil(n / 2)


def add_pose_level_rank(df, ligand_cols):
    out = df.copy(); out["score_rank"] = False
    for _, ix in out.groupby(ligand_cols).indices.items():
        out.iloc[ix, out.columns.get_loc("score_rank")] = score_rank_pass(out.iloc[ix]["vina_score"].values)
    return out


def representative(df, ligand_cols):
    d = df.sort_values(ligand_cols + ["microstate_id", "vina_score", "conf_rank"])
    return d.drop_duplicates(ligand_cols + ["microstate_id"], keep="first").reset_index(drop=True)


def ligand_table(rep, ligand_cols, default_col="is_default"):
    rep = rep.copy(); rep["score_rank"] = False
    for _, ix in rep.groupby(ligand_cols).indices.items():
        rep.iloc[ix, rep.columns.get_loc("score_rank")] = score_rank_pass(rep.iloc[ix]["vina_score"].values)
    rows = []
    for key, g in rep.groupby(ligand_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        d = g[g[default_col]]
        n_micro = len(g)
        best = g.sort_values(["vina_score", "microstate_id"]).iloc[0]
        r = dict(zip(ligand_cols, key)); r.update(n_micro=n_micro, has_default=len(d) == 1, best_microstate=best.microstate_id, best_score=best.vina_score)
        if len(d) == 1:
            dd = d.iloc[0]
            r.update(default_microstate=dd.microstate_id, default_score=dd.vina_score, best_is_default=bool(best.microstate_id == dd.microstate_id), gain=float(dd.vina_score - best.vina_score))
            for c in CRITERIA:
                r["default_" + c] = bool(dd[c]); r["best_" + c] = bool(best[c])
                r["flip_" + c] = bool(dd[c]) != bool(best[c])
                r["rescue_" + c] = (not bool(dd[c])) and bool(best[c]); r["degrade_" + c] = bool(dd[c]) and not bool(best[c])
        rows.append(r)
    return pd.DataFrame(rows)


def flip_rates(lt, group_cols, denominators=("primary", "best_differs")):
    out = []
    prim = lt[lt.has_default.astype(bool) & (lt.n_micro >= 2)].copy()
    for c in ["best_is_default"] + [p_ + k for p_ in ("flip_", "rescue_", "degrade_") for k in CRITERIA]:
        prim[c] = prim[c].astype(bool)
    for key, g in prim.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for c in CRITERIA:
            for den, gg in (("primary", g), ("best_differs", g[~g.best_is_default])):
                n = len(gg); k = int(gg["flip_" + c].sum()); lo, hi = wilson(k, n)
                out.append(dict(zip(group_cols, key), criterion=c, denominator=den, n_ligands=n, n_flip=k, flip_pct=100 * k / n if n else float("nan"), wilson_lo=100 * lo, wilson_hi=100 * hi,
                                n_rescue=int(gg["rescue_" + c].sum()), n_degrade=int(gg["degrade_" + c].sum())))
    return pd.DataFrame(out)


def cohen_kappa_counts(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return float("nan")
    po = (a + d) / n; pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def kappa_cluster(df, x, y, cluster_cols, n_boot, rng):
    d = df[cluster_cols + [x, y]].copy()
    d["a"] = (d[x] & d[y]).astype(int); d["b"] = (d[x] & ~d[y]).astype(int); d["c"] = (~d[x] & d[y]).astype(int); d["d"] = (~d[x] & ~d[y]).astype(int)
    g = d.groupby(cluster_cols)[["a", "b", "c", "d"]].sum().values
    tot = g.sum(0); k0 = cohen_kappa_counts(*tot)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s = g[idx].sum(1)
    ks = np.array([cohen_kappa_counts(*r) for r in s])
    lo, hi = np.nanpercentile(ks, [2.5, 97.5]) if np.isfinite(ks).any() else (float("nan"), float("nan"))
    return dict(n_poses=int(tot.sum()), n_clusters=len(g), kappa=k0, kappa_lo=float(lo), kappa_hi=float(hi), p_both_true=float(tot[0] / tot.sum()), p_first_true=float((tot[0] + tot[1]) / tot.sum()), p_second_true=float((tot[0] + tot[2]) / tot.sum()),
                agree=float((tot[0] + tot[3]) / tot.sum()))


def two_proportion_z(k1, n1, k2, n2):
    from scipy import stats
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (k1 / n1 - k2 / n2) / se
    return z, float(2 * stats.norm.sf(abs(z)))

_DESC = {
    "dataset": ("", "ten_target_main / ten_target_budget6 = original runs re-checked; dude_full / litpcba / zinc_unmatched / zinc_charge_mismatch = decoy-matching ligand sets"),
    "set": ("", "decoy-matching ligand set"), "seed": ("", "docking seed of the original run (directory name); ALL = pooled"), "target": ("", "receptor target; ALL = pooled over targets"),
    "role": ("", "active / negative; ALL = both"), "round": ("", "0 = active, k = k-th negative of the active"), "ligand_idx": ("", "row index of the ligand in the original run"),
    "ligand_id": ("", "database id"), "uid": ("", "md5(target|smiles)[:12], key of the decoy-matching docking records"), "compound_id": ("", "original ledger compound id"),
    "label": ("", "1 = active, 0 = decoy (original run)"), "source": ("", "original run ligand source (active / decoy)"),
    "n_micro": ("", "microstates with a docked pose (and a PoseBusters verdict)"), "n_microstates_failed": ("", "eligible microstates whose docking failed"),
    "has_default": ("", "the default microstate was docked"), "default_microstate": ("", "default microstate id (original: first candidate of the single strategy; decoy-matching: order 0)"),
    "best_microstate": ("", "microstate of the lowest Vina score among the docked microstates"), "default_score": ("kcal/mol", "Vina score of the default microstate's representative pose"),
    "best_score": ("kcal/mol", "lowest Vina score of the ligand"), "best_is_default": ("", "best-scoring microstate equals the default microstate"), "gain": ("kcal/mol", "default_score - best_score (>= 0)"),
    "criterion": ("", "geom | full_pb | score_rank"), "denominator": ("", "primary = ligands with a default and >= 2 microstates; best_differs = those whose best microstate is not the default"),
    "n_ligands": ("", "ligands in the denominator"), "n_flip": ("", "ligands whose default and best microstate differ in the criterion"), "flip_pct": ("%", "100 * n_flip / n_ligands"),
    "wilson_lo": ("%", "Wilson 95% lower bound of flip_pct"), "wilson_hi": ("%", "Wilson 95% upper bound of flip_pct"), "n_rescue": ("", "default invalid -> best valid"), "n_degrade": ("", "default valid -> best invalid"),
    "unit": ("", "all_poses | representative_poses"), "criterion_a": ("", "first criterion"), "criterion_b": ("", "second criterion"), "n_poses": ("", "poses in the table"), "n_clusters": ("", "ligands (bootstrap clusters)"),
    "kappa": ("", "Cohen's kappa"), "kappa_lo": ("", "ligand-cluster bootstrap 2.5%"), "kappa_hi": ("", "ligand-cluster bootstrap 97.5%"), "p_both_true": ("", "share of poses valid under both criteria"),
    "p_first_true": ("", "share valid under the first criterion"), "p_second_true": ("", "share valid under the second criterion"), "agree": ("", "observed agreement"),
    "n_ligands_checked": ("", "ten-target docking seed 20260210 ligands with at least one checked strategy pose"), "n_flip_old_definition": ("", "ligands with >= 2 distinct strategy-chosen microstates, one valid and one invalid (original p2 definition, full_pb)"),
    "flip_pct_old_definition": ("%", "100 * n_flip_old_definition / n_ligands_checked"), "n_ligands_new_def": ("", "n_ligands of the new definition (primary denominator, full_pb)"), "n_flip_new_def": ("", "n_flip of the new definition"),
    "flip_pct_new_def": ("%", "flip_pct of the new definition"),
}
_PREFIX = {"flip_": ("", "default and best microstate differ in validity under {c}"), "rescue_": ("", "default invalid and best valid under {c}"), "degrade_": ("", "default valid and best invalid under {c}"),
           "default_": ("", "verdict of the default microstate's representative pose under {c}"), "best_": ("", "verdict of the best-scoring microstate's representative pose under {c}")}


def column_info(c):
    if c in _DESC:
        return _DESC[c]
    for pre, (u, d) in _PREFIX.items():
        if c.startswith(pre) and c[len(pre):] in CRITERIA:
            return (u, d.format(c=c[len(pre):]))
    raise KeyError("no description for column " + c)


def write_schemas(P, tables, block):
    import csv
    for path, grain, desc in tables:
        header = next(csv.reader(open(path, newline="", encoding="utf-8")))
        P.write_schema(path, [(c,) + column_info(c) for c in header], grain, desc, extra=dict(provenance=block))
