import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
VF = Path(os.environ.get("DOCKING_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RAW, RES = OUT_ROOT / "raw_outputs/decoy_matching", OUT_ROOT / "results/decoy_matching"; RES.mkdir(parents=True, exist_ok=True)
A = CFG["comparison"]
ALIAS = {"dominant_net_charge": "dominant_charge", "n_ionizable_sites": "n_sites", "n_microstates": "n_microstates", "n_tautomers": "n_tautomers", "n_rotatable_bonds": "n_rot", "n_heavy_atoms": "n_heavy"}
assert set(A["variables"]) == set(ALIAS), "comparison variables in the config differ from the mapping: %s" % sorted(set(A["variables"]) ^ set(ALIAS))
CAT, CON = [ALIAS[v] for v in A["categorical"]], [ALIAS[v] for v in A["continuous"]]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); out = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); out[i] = min(1.0, run)
    return out


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.rankdata(np.concatenate([a, b]))
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return 2.0 * u / (len(a) * len(b)) - 1.0


def charge_class(q):
    return np.where(q <= -2, "<=-2", np.where(q >= 2, ">=+2", q.astype(int).astype(str)))


def prep(df):
    df = df[df.ok.astype(str) == "True"].copy()
    df["dominant_charge"] = charge_class(df["dominant_charge"].astype(float))
    df["n_sites"] = np.minimum(df["n_sites"].astype(float), 3).astype(int).astype(str)
    for c in CON:
        df[c] = df[c].astype(float)
    return df


def chi2_v(x_act, x_neg):
    cats = sorted(set(x_act) | set(x_neg))
    tab = np.array([[sum(1 for v in x_act if v == c) for c in cats], [sum(1 for v in x_neg if v == c) for c in cats]], float)
    while tab.shape[1] > 2:
        exp = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
        bad = np.where((exp < 1).any(0))[0]
        if not len(bad):
            break
        j = bad[0]; k = j + 1 if j + 1 < tab.shape[1] else j - 1
        tab[:, k] += tab[:, j]; tab = np.delete(tab, j, axis=1)
    if tab.shape[1] < 2 or (tab.sum(0) == 0).all():
        return float("nan"), float("nan"), float("nan"), tab.shape[1]
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    n = tab.sum(); v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if n else float("nan")
    return float(chi2), float(p), v, tab.shape[1]


def comparison_pair(act, neg, dataset, target, extra=None):
    rows = []
    for v in CAT:
        chi2, p, V, k = chi2_v(list(act[v]), list(neg[v]))
        rows.append(dict(dataset=dataset, target=target, variable=v, kind="categorical", test="chi2", statistic=chi2, p=p, effect_size=V, effect_name="cramers_v", n_active=len(act), n_negative=len(neg),
                         median_active=None, median_negative=None, n_classes=k))
    for v in CON:
        a, b = act[v].values, neg[v].values
        ks = stats.ks_2samp(a, b)
        rows.append(dict(dataset=dataset, target=target, variable=v, kind="continuous", test="ks", statistic=float(ks.statistic), p=float(ks.pvalue), effect_size=cliffs_delta(a, b), effect_name="cliffs_delta",
                         n_active=len(a), n_negative=len(b), median_active=float(np.median(a)), median_negative=float(np.median(b)), n_classes=None))
    return rows


def load_props(ds):
    f = RAW / ("ligand_properties_%s.csv" % ds)
    return pd.read_csv(f, dtype={"ligand_id": str, "smiles": str}, keep_default_na=False, na_values=[""])


COLS = [("dataset", "", "dude / litpcba / zinc_unmatched / zinc_charge_mismatch"), ("target", "", "target"), ("variable", "", "dominant_charge, n_sites, n_microstates, n_tautomers, n_rot, n_heavy"),
        ("kind", "", "categorical / continuous"), ("test", "", "chi2 (categorical) or ks (continuous)"), ("statistic", "", "chi-square statistic or KS D"), ("p", "", "p value"),
        ("effect_size", "", "Cramer's V or Cliff's delta (> 0: actives larger)"), ("effect_name", "", "cramers_v / cliffs_delta"), ("n_active", "", "ligands with a successful enumeration among the actives"),
        ("n_negative", "", "same among the negatives"), ("median_active", "", "median (continuous variables)"), ("median_negative", "", "median (continuous variables)"), ("n_classes", "", "classes used in the chi-square test after pooling expected counts < 1"),
        ("holm_p", "", "Holm-adjusted p over targets x variables within the dataset"), ("significant_holm", "", "holm_p < alpha")]
summ = dict(datasets={}, notes=[])
t0 = time.time()
allrows = {}
d = load_props("dude")
for ds, kinds in (("dude", ("active", "decoy")),):
    rows = []
    for t in CFG["targets"]["dude"]:
        sub = d[d.target == t]
        act, neg = prep(sub[sub.kind == kinds[0]]), prep(sub[sub.kind == kinds[1]])
        rows += comparison_pair(act, neg, ds, t)
    allrows[ds] = rows
    summ["datasets"][ds] = dict(n_failed_enum=int((d.ok.astype(str) != "True").sum()), n_ligands=len(d))
l = load_props("litpcba")
rows = []
skipped = []
for t in CFG["targets"]["litpcba"]:
    sub = l[l.target == "litpcba_" + t] if "litpcba_" + t in set(l.target) else l[l.target == t]
    if not (sub.kind == "active").any() or not (sub.kind == "inactive").any():
        skipped.append(t); continue
    act, neg = prep(sub[sub.kind == "active"]), prep(sub[sub.kind == "inactive"])
    rows += comparison_pair(act, neg, "litpcba", sub.target.iloc[0])
allrows["litpcba"] = rows
summ["datasets"]["litpcba"] = dict(n_failed_enum=int((l.ok.astype(str) != "True").sum()), n_ligands=len(l))
summ["litpcba_targets_skipped"] = skipped
if skipped:
    summ["notes"].append("LIT-PCBA targets skipped (property parts missing): %s" % skipped)
zf = RAW / "ligand_properties_zinc_drawn.csv"
if zf.exists():
    z = pd.read_csv(zf, dtype={"ligand_id": str}, keep_default_na=False, na_values=[""])
    for ds in ("zinc_unmatched", "zinc_charge_mismatch"):
        rows = []
        for t in CFG["targets"]["dude"]:
            act = prep(d[(d.target == t) & (d.kind == "active")])
            neg = prep(z[(z.set == ds) & (z.target == t)])
            rows += comparison_pair(act, neg, ds, t)
        allrows[ds] = rows
for ds, rows in allrows.items():
    h = holm([r["p"] for r in rows])
    for r, hp in zip(rows, h):
        r["holm_p"] = hp; r["significant_holm"] = bool(hp < CFG["tests"]["alpha"])
    out = RES / ("property_comparison_%s.csv" % ds)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([c[0] for c in COLS])
        for r in rows:
            w.writerow(["" if r.get(c[0]) is None or (isinstance(r.get(c[0]), float) and not np.isfinite(r.get(c[0]))) else (format(r[c[0]], ".8g") if isinstance(r.get(c[0]), float) else r.get(c[0])) for c in COLS])
    BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RAW / "ligand_properties_dude.csv", RAW / "ligand_properties_litpcba.csv"], seed=None, extra=dict(dataset=ds))
    P.write_schema(out, COLS, "one row per target x variable", "Property-matching comparison of actives vs negatives (%s)" % ds, extra=dict(provenance=BLOCK))
    df = pd.DataFrame(rows)
    summ["datasets"].setdefault(ds, {}).update(n_tests=len(rows), n_sig_holm=int(df.significant_holm.sum()),
                                               sig_by_variable={v: int(df[df.variable == v].significant_holm.sum()) for v in CAT + CON},
                                               median_abs_effect={v: float(df[df.variable == v].effect_size.abs().median()) for v in CAT + CON},
                                               max_abs_effect={v: float(df[df.variable == v].effect_size.abs().max()) for v in CAT + CON})
rows = []
first_matched = {}
for t in CFG["targets"]["dude"]:
    dec = d[(d.target == t) & (d.kind == "decoy")].reset_index(drop=True)
    ism = [(ln.split()[0], ln.split()[1]) for ln in open(Path(CFG["paths"]["dude_inputs"].replace("$DOCKING_ROOT", str(VF))) / t / "decoys_final.ism") if len(ln.split()) >= 2]
    first = set(ism[:A["first_n_lines_check"]["decoys"]])
    inf = np.array([k in first for k in zip(dec.smiles, dec.ligand_id)], dtype=bool)
    first_matched[t] = dict(lines=int(A["first_n_lines_check"]["decoys"]), distinct_pairs=len(first), rows_matched=int(inf.sum()))
    a = prep(dec[inf]); b = prep(dec[~inf])
    rows += comparison_pair(a, b, "first900_vs_rest", t)
h = holm([r["p"] for r in rows])
for r, hp in zip(rows, h):
    r["holm_p"] = hp; r["significant_holm"] = bool(hp < CFG["tests"]["alpha"])
out = RES / "property_comparison_first900_decoys.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow([c[0] for c in COLS])
    for r in rows:
        w.writerow(["" if r.get(c[0]) is None or (isinstance(r.get(c[0]), float) and not np.isfinite(r.get(c[0]))) else (format(r[c[0]], ".8g") if isinstance(r.get(c[0]), float) else r.get(c[0])) for c in COLS])
P.write_schema(out, COLS, "one row per target x variable", "First 900 lines of decoys_final.ism (original run pool) vs the remaining decoys of the target", extra=dict(provenance=P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RAW / "ligand_properties_dude.csv"], seed=None)))
df = pd.DataFrame(rows)
summ["first900_vs_rest"] = dict(matching="(smiles, ligand_id)", matched_by_target=first_matched, n_tests=len(rows), n_sig_holm=int(df.significant_holm.sum()), max_abs_effect={v: float(df[df.variable == v].effect_size.abs().max()) for v in CAT + CON})
summ["wall_seconds"] = round(time.time() - t0, 1)
json.dump(summ, open(RES / "property_comparison_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summ, indent=1)[:3500])
