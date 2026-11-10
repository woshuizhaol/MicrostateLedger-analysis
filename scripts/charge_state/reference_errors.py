import json, sys, time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C
import charge_schemas as S

A, ST, RW = C.CFG["anchor"], C.CFG["stats"], C.CFG["reweight"]
SCRIPT = Path(__file__).resolve()
METHODS = A["methods"]
PRIM = A["primary_set"]
C.check_pins()
anch_path = C.RES / "anchor_set.csv"
df = pd.read_csv(anch_path)
for c in ("complete_profile", "complete_relaxed", "ambiguous", "coupled_sites"):
    df[c] = df[c].astype(bool)
df["q_neutral_input"] = 0
CLSNAMES = ["<=-2", "-1", "0", "+1", ">=+2"]


def cls5(q):
    return "<=-2" if q <= -2 else "-1" if q == -1 else "0" if q == 0 else "+1" if q == 1 else ">=+2"


BASE = {"complete": df[df.complete_profile], "complete_relaxed": df[df.complete_relaxed], "sensitivity": df[~df.complete_profile], "all": df}
VAR = {
    "excl_ambiguous": lambda d: d[~d.ambiguous],
    "excl_coupled": lambda d: d[~d.coupled_sites],
    "IUPAC": lambda d: d[d.source == "IUPAC"],
    "SAMPL": lambda d: d[d.source == "SAMPL"],
    "ChEMBL": lambda d: d[d.source == "ChEMBL"],
    "n_sites_1": lambda d: d[d.n_sites_pred <= 1],
    "n_sites_ge2": lambda d: d[d.n_sites_pred >= 2],
    "non_overlap_molgpka_example": lambda d: d[~d.ml_in_molgpka_example_sdf.astype(bool)],
    "non_overlap_pkasolver_train": lambda d: d[~d.ml_in_pkasolver_train.astype(bool)],
    "non_overlap_chembl37": lambda d: d[~d.ml_in_chembl37.astype(bool)],
}
SETS = {}
for b, d in BASE.items():
    SETS[b] = d
    if b in A["variant_base_sets"]:
        for v, f in VAR.items():
            SETS["%s:%s" % (b, v)] = f(d)


def rate_row(name, method, d):
    q = d["q_" + method]
    ok = q.notna()
    n = int(ok.sum()); k = int((q[ok].astype(int) != d.loc[ok, "exp_dominant_charge"].astype(int)).sum())
    lo, hi = C.wilson(k, n)
    return dict(set=name, method=method, k_wrong=k, n=n, error_rate=k / n if n else None, wilson_lo=lo, wilson_hi=hi, n_set=len(d), n_failed=len(d) - n)


def class_of_dude(smi):
    m = C.clean_mol(Chem.MolFromSmiles(smi))
    if m is None:
        return None, None, None
    sc = C.site_counts(m, Chem.MolToSmiles(m))
    return sc["primary_class"], m.GetNumHeavyAtoms(), max(sc["n_dd_strict"], sc["n_ind_strict"])


def main():
    t0 = time.time()
    err_rows = [rate_row(n, m, d) for n, d in SETS.items() for m in METHODS]
    p_err = C.RES / "anchor_method_error.csv"
    C.write_csv(p_err, S.METHOD_ERR, err_rows)
    errt = pd.DataFrame(err_rows).set_index(["set", "method"])

    cls_rows = []
    for sname in A["class_report_sets"]:
        d = SETS[sname]
        for m in METHODS:
            for c in C.PRIORITY:
                dc = d[d.functional_class == c]
                q = dc["q_" + m]; ok = q.notna(); n = int(ok.sum())
                k = int((q[ok].astype(int) != dc.loc[ok, "exp_dominant_charge"].astype(int)).sum())
                lo, hi = C.wilson(k, n)
                cls_rows.append(dict(set=sname, method=m, functional_class=c, k_wrong=k, n=n, error_rate=k / n if n else None,
                                     wilson_lo=lo if n else None, wilson_hi=hi if n else None, low_n=n < C.CFG["classes"]["min_n_for_class_error"]))
    p_cls = C.RES / "anchor_error_by_class.csv"
    C.write_csv(p_cls, S.BY_CLASS, cls_rows)

    conf_rows = []
    for sname in A["class_report_sets"]:
        d = SETS[sname]
        for m in METHODS:
            q = d["q_" + m]; ok = q.notna()
            e = d.loc[ok, "exp_dominant_charge"].astype(int).map(cls5); g = q[ok].astype(int).map(cls5)
            ct = pd.crosstab(e, g)
            for ec in ct.index:
                for gc in ct.columns:
                    if ct.loc[ec, gc]:
                        conf_rows.append(dict(set=sname, method=m, exp_charge_class=ec, method_charge_class=gc, count=int(ct.loc[ec, gc])))
    p_conf = C.RES / "anchor_confusion.csv"
    C.write_csv(p_conf, S.ANCH_CONF, conf_rows)

    dude = pd.read_csv(C.PATHS["dude_tsv"], sep="\t")
    with mp.get_context("fork").Pool(ST["threads"]) as pool:
        res = pool.map(class_of_dude, dude.parent_smiles.tolist(), chunksize=16)
    dude["functional_class"] = [r[0] for r in res]; dude["heavy"] = [r[1] for r in res]; dude["n_sites_pred"] = [r[2] for r in res]
    assert dude.functional_class.notna().all(), "unparsable DUD-E ligand in the composition step"
    STRATA = {"functional_class": list(C.PRIORITY), "n_sites_stratum[post_hoc]": ["<=1_group", ">=2_groups"]}
    dude["n_sites_stratum[post_hoc]"] = np.where(dude.n_sites_pred <= 1, "<=1_group", ">=2_groups")
    df["n_sites_stratum[post_hoc]"] = np.where(df.n_sites_pred <= 1, "<=1_group", ">=2_groups")
    for k_ in BASE:
        BASE[k_] = df.loc[BASE[k_].index]
        SETS[k_] = BASE[k_]
    comp_rows, comp_c = [], {}
    for scheme, strata in STRATA.items():
        vc = dude[scheme].value_counts()
        comp_c[scheme] = {c: int(vc.get(c, 0)) for c in strata}
        n_anchor_c = BASE["complete"][scheme if scheme != "functional_class" else "functional_class"].value_counts()
        for c in strata:
            comp_rows.append(dict(scheme=scheme, stratum=c, n_dude=comp_c[scheme][c], weight=comp_c[scheme][c] / len(dude), n_anchor_complete=int(n_anchor_c.get(c, 0)), covered=int(n_anchor_c.get(c, 0)) > 0))
    p_comp = C.RES / "dude_class_composition.csv"
    C.write_csv(p_comp, S.DUDE_CLASS, comp_rows)

    rng = np.random.default_rng(ST["seed"] + 31)
    B = ST["n_boot"]
    rw_rows = []
    for scheme, strata in STRATA.items():
        w = np.array([comp_c[scheme][c] for c in strata], float); w /= w.sum()
        for sname in A["class_report_sets"]:
            d = SETS[sname]
            for m in METHODS:
                n_c, e_c = np.zeros(len(strata)), np.zeros(len(strata))
                for i, c in enumerate(strata):
                    dc = d[d[scheme] == c]; q = dc["q_" + m]; ok = q.notna()
                    n_c[i] = ok.sum()
                    e_c[i] = (q[ok].astype(int) != dc.loc[ok, "exp_dominant_charge"].astype(int)).mean() if ok.sum() else 0.0
                cov = n_c > 0
                point = float((w[cov] * e_c[cov]).sum() / w[cov].sum())
                wb = rng.multinomial(len(dude), w, size=B) / len(dude)
                kb = np.zeros((B, len(strata)))
                for i in range(len(strata)):
                    if cov[i]:
                        kb[:, i] = rng.binomial(int(n_c[i]), e_c[i], size=B) / n_c[i]
                wbc = wb * cov
                boot = (wbc * kb).sum(1) / wbc.sum(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rw_rows.append(dict(scheme=scheme, anchor_set=sname, method=m, expected_error=point, lo=lo, hi=hi, covered_weight=float(w[cov].sum()),
                                    unweighted_error=errt.loc[(sname, m), "error_rate"],
                                    class_weights="|".join("%s:%.4f" % (c, w[i]) for i, c in enumerate(strata) if cov[i]),
                                    label=RW["extrapolation_label"] + (" (post hoc: added after the first error table was seen)" if "post_hoc" in scheme else "")))
    p_rw = C.RES / "dude_reweighted_error.csv"
    C.write_csv(p_rw, S.REWEIGHT, rw_rows)

    leak = dict(note=C.CFG["leakage"]["note"], primary_set=PRIM, resources={})
    for res_name, col in (("molgpka_example", "ml_in_molgpka_example_sdf"), ("pkasolver_train", "ml_in_pkasolver_train"), ("chembl37", "ml_in_chembl37")):
        d = BASE[PRIM]; ov = d[d[col].astype(bool)]
        sub = {}
        for m in METHODS:
            sub[m] = dict(all=errt.loc[(PRIM, m), "error_rate"], non_overlap=errt.loc[("%s:non_overlap_%s" % (PRIM, res_name), m), "error_rate"],
                          non_overlap_wilson=[errt.loc[("%s:non_overlap_%s" % (PRIM, res_name), m), "wilson_lo"], errt.loc[("%s:non_overlap_%s" % (PRIM, res_name), m), "wilson_hi"]])
        ovm = rate_row("overlap", "ml_pka", ov) if len(ov) else None
        leak["resources"][res_name] = dict(n_complete=len(d), n_overlap=len(ov), n_non_overlap=len(d) - len(ov), ml_pka_error_on_overlap=(ovm["error_rate"] if ovm else None),
                                           error_rates=sub)
    json.dump(leak, open(C.RES / "ml_leakage_check.json", "w"), indent=2, ensure_ascii=False)

    dd = BASE[PRIM]
    summary = dict(
        primary_set=PRIM, n_primary=len(dd), n_all=len(df),
        headline={m: dict(k=int(errt.loc[(PRIM, m), "k_wrong"]), n=int(errt.loc[(PRIM, m), "n"]), rate=errt.loc[(PRIM, m), "error_rate"],
                          lo=errt.loc[(PRIM, m), "wilson_lo"], hi=errt.loc[(PRIM, m), "wilson_hi"]) for m in METHODS},
        representativeness=dict(anchor_complete_heavy_atoms_median=float(dd.heavy_atoms.median()), anchor_complete_mw_median=float(dd.mw.median()),
                                dude_heavy_atoms_median=float(dude.heavy.median()),
                                anchor_complete_n_sites_ind_ge2=float((dd.n_sites_ind >= 2).mean()),
                                exp_charged_fraction_complete=float((dd.exp_dominant_charge != 0).mean())),
        dude_class_composition=comp_c, wall_seconds=round(time.time() - t0, 1))
    ob_ver = __import__("subprocess").run([str(C.PATHS["obabel"]), "-V"], capture_output=True, text=True).stdout.strip()
    BLOCK = C.P.provenance_block(SCRIPT, C.ROOT / "configs/charge_state.yaml", [anch_path, C.PATHS["dude_tsv"]], seed=ST["seed"], n_boot=B,
                                 extra=dict(openbabel=ob_ver), tools_extra=("rdkit",))
    summary["provenance"] = BLOCK
    grains = S.ROW_GRAIN
    C.P.write_schema(p_err, S.METHOD_ERR, grains["anchor_method_error.csv"], "Method-vs-experiment dominant-charge error rates with Wilson CIs on the anchor sets and their variants", extra=dict(provenance=BLOCK))
    C.P.write_schema(p_cls, S.BY_CLASS, grains["anchor_error_by_class.csv"], "Error rates by primary functional class (empty classes with n = 0)", extra=dict(provenance=BLOCK))
    C.P.write_schema(p_conf, S.ANCH_CONF, grains["anchor_confusion.csv"], "Confusion between experimental and method charge class on the anchor sets", extra=dict(provenance=BLOCK))
    C.P.write_schema(p_comp, S.DUDE_CLASS, grains["dude_class_composition.csv"], "Functional-class composition of the 3,584 DUD-E actives and anchor coverage", extra=dict(provenance=BLOCK))
    C.P.write_schema(p_rw, S.REWEIGHT, grains["dude_reweighted_error.csv"], "Expected method error under the DUD-E class composition (weighted extrapolation)", extra=dict(provenance=BLOCK))
    json.dump(summary, open(C.RES / "reference_summary.json", "w"), indent=2, ensure_ascii=False)
    C.log("done; primary set n =", len(dd))


if __name__ == "__main__":
    main()
