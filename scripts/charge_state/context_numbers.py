import json, sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C

M6 = C.CFG["anchor"]["methods"]
a = pd.read_csv(C.RES / "anchor_set.csv"); a["q_neutral_input"] = 0
for c in ("complete_profile",): a[c] = a[c].astype(bool)
refp = C.ROOT / "inputs/data/iupac_pka_detail.tsv"
original = pd.read_csv(refp, sep="\t")
original["skel"] = [C.skeleton_key(Chem.MolFromSmiles(s)) for s in original.smiles]
e = original.exp_pka.astype(float)


def wrong(d, m):
    q = d["q_" + m]; ok = q.notna()
    return int((q[ok].astype(int) != d.loc[ok, "exp_dominant_charge"].astype(int)).sum()), int(ok.sum())


def rates(d):
    out = {}
    for m in M6:
        k, n = wrong(d, m)
        out[m] = dict(k=k, n=n, rate=k / n if n else None, wilson=list(C.wilson(k, n)))
    return out


ov = original.merge(a, left_on="skel", right_on="ik_skeleton", how="inner")
res = dict(
    original_set=dict(n_rows=len(original), n_with_prediction=int(original.pred_pka_closest.notna().sum()), share_exp_pka_below_7p4=float((e < 7.4).mean()), median_exp_pka=float(e.median()),
                 share_within_2_of_7p4=float(((e - 7.4).abs() < 2).mean()), n_found_in_anchor=len(ov), n_found_in_complete=int(ov.complete_profile.sum()),
                 share_exp_charged_on_overlap=float((ov.exp_dominant_charge != 0).mean()), errors_on_overlap_all=rates(ov), errors_on_overlap_complete=rates(ov[ov.complete_profile]),
                 original_file=str(refp), original_file_md5=C.P.file_md5(refp)),
    zero_site=dict(n_complete=int(a.complete_profile.sum()), n_complete_zero_sites=int(((a.n_sites_pred == 0) & a.complete_profile).sum()),
                   n_zero_sites_exp_charged=int(((a.n_sites_pred == 0) & a.complete_profile & (a.exp_dominant_charge != 0)).sum()),
                   errors_complete_without_zero_sites=rates(a[a.complete_profile & (a.n_sites_pred >= 1)])),
    provenance=C.P.provenance_block(Path(__file__), C.ROOT / "configs/charge_state.yaml", [C.RES / "anchor_set.csv", refp], extra=dict(note="descriptive context numbers; not independently re-derived by the consistency check")))
cs = a[a.complete_profile].copy()
kc = pd.read_csv(C.RES / "anchor_constants.csv")
cs["ladder_type"] = cs.mol_id.map(kc.groupby("mol_id").direction.agg(lambda x: "+".join(sorted(set(x)))))
res["by_ladder_type"] = {lt: dict(n=len(g), **{m: wrong(g, m)[0] / wrong(g, m)[1] for m in M6}) for lt, g in cs.groupby("ladder_type")}
res["ladder_type_note"] = "A = only acid constants, AH = only conjugate-acid (base) constants, A+AH = both (amphoteric)"
res["signed_error"] = {m: {str(k): int(v) for k, v in (cs["q_" + m] - cs.exp_dominant_charge).value_counts().sort_index().items()} for m in M6}
res["signed_error_note"] = "method net charge minus experimental dominant net charge, complete set; 0 = correct"
original_sk = set(ov.skel)
rest = cs[~cs.ik_skeleton.isin(original_sk)]
res["complete_without_old_set_molecules"] = dict(n_removed=len(cs) - len(rest), errors=rates(rest))
res["exp_neutral_vs_charged_dimorphite"] = dict(n_exp_neutral=int((cs.exp_dominant_charge == 0).sum()), dimorphite_charged_on_exp_neutral=int(((cs.exp_dominant_charge == 0) & (cs.q_dimorphite_dom != 0)).sum()),
                                                n_exp_charged=int((cs.exp_dominant_charge != 0).sum()), dimorphite_neutral_on_exp_charged=int(((cs.exp_dominant_charge != 0) & (cs.q_dimorphite_dom == 0)).sum()))
clean = ~a.had_dropped_constant.astype(bool) & ~a.ladder_gap.astype(bool) & (a.n_pka_exp_window >= 1)
res["gate_sensitivity"] = {
    "joint_max_of_both (primary)": dict(n=int(a.complete_profile.sum()), errors=rates(a[a.complete_profile])),
    "independent_smarts_only": dict(n=int((clean & (a.n_pka_exp_window >= a.n_sites_ind)).sum()), errors=rates(a[clean & (a.n_pka_exp_window >= a.n_sites_ind)])),
    "dimorphite_smarts_only": dict(n=int((clean & (a.n_pka_exp_window >= a.n_sites_dd)).sum()), errors=rates(a[clean & (a.n_pka_exp_window >= a.n_sites_dd)])),
    "no_site_gate_clean_only": dict(n=int(clean.sum()), errors=rates(a[clean])),
}
margin = C.CFG["post_freeze"]["sampl_type_check_margin"]
_win = {}


def _wins(smi, kind):
    if (smi, kind) not in _win:
        _win[(smi, kind)] = [(x[3] - margin, x[4] + margin) for x in C.independent_sites(Chem.MolFromSmiles(smi)) if x[1] == kind]
    return _win[(smi, kind)]


kk = kc.merge(a[["mol_id", "smiles", "complete_profile"]], on="mol_id")
kk["unexplained"] = [not any(lo <= p_ <= hi for lo, hi in _wins(sm, "base" if d_ == "AH" else "acid")) for sm, d_, p_ in zip(kk.smiles, kk.direction, kk.pka)]
flag_ids = set(kk.loc[kk.unexplained, "mol_id"])
cs["has_unexplained"] = cs.mol_id.isin(flag_ids)
res["unexplained_constants"] = dict(
    note="POST HOC coverage diagnostic; defined by our own independent SMARTS windows (+/- %.1f), see comment in context_numbers.py" % margin,
    n_constants=len(kk), n_unexplained=int(kk.unexplained.sum()),
    by_source={s_: dict(n=int((kk.source == s_).sum()), unexplained=int(kk.loc[kk.source == s_, "unexplained"].sum())) for s_ in sorted(kk.source.unique())},
    complete_flagged=dict(n=int(cs.has_unexplained.sum()), share=float(cs.has_unexplained.mean()), errors=rates(cs[cs.has_unexplained])),
    complete_rest=dict(n=int((~cs.has_unexplained).sum()), errors=rates(cs[~cs.has_unexplained])))
res["complete_without_amphoteric"] = dict(n_removed=int((cs.ladder_type == "A+AH").sum()), n=int((cs.ladder_type != "A+AH").sum()), errors=rates(cs[cs.ladder_type != "A+AH"]),
                                          note="ladder_type A+AH = the anchor molecule has both acid (A) and conjugate-acid (AH) constants")
fix_ids = ["A0276", "A0292", "A1098"]
alt = a.copy()
changes = {}
for mid in fix_ids:
    g = kc[kc.mol_id == mid]
    consts = [("A" if (d_ == "AH" and p_ < 5.5) else d_, p_) for d_, p_ in zip(g.direction, g.pka)]
    q_new, pop_new = C.dominant_of(C.hh_pmf(consts))
    changes[mid] = dict(smiles=str(a.loc[a.mol_id == mid, "smiles"].iloc[0]), constants_as_labelled=[[d_, float(p_)] for d_, p_ in zip(g.direction, g.pka)],
                        q_as_labelled=int(a.loc[a.mol_id == mid, "exp_dominant_charge"].iloc[0]), q_corrected=int(q_new), pop_corrected=float(pop_new))
    alt.loc[alt.mol_id == mid, "exp_dominant_charge"] = q_new
res["iupac_label_sensitivity"] = dict(
    note="source labelling error (carboxylic acid typed pKaH) in 3 complete-set molecules; found via an SMARTS screen which is not exhaustive",
    changes=changes, complete_as_labelled=rates(a[a.complete_profile]), complete_corrected=rates(alt[alt.complete_profile]))
json.dump(res, open(C.RES / "context_original_set.json", "w"), indent=2, ensure_ascii=False)
o = res["original_set"]
print("original set: n_found_in_anchor", o["n_found_in_anchor"], "complete", o["n_found_in_complete"], "| exp charged share", round(o["share_exp_charged_on_overlap"], 3))
for k in ("errors_on_overlap_all", "errors_on_overlap_complete"):
    print(k, {m: "%d/%d=%.1f%%" % (v["k"], v["n"], 100 * v["rate"]) for m, v in o[k].items()})
z = res["zero_site"]; print("zero-site:", z["n_complete_zero_sites"], "of", z["n_complete"], "| exp charged:", z["n_zero_sites_exp_charged"])
print({m: "%.1f%%" % (100 * v["rate"]) for m, v in z["errors_complete_without_zero_sites"].items()})
