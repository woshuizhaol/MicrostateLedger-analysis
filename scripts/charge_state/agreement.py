import csv, json, subprocess, sys, time
import multiprocessing as mp
from importlib import metadata
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C
import charge_schemas as S

CFGA, ST = C.CFG["agreement"], C.CFG["stats"]
METHODS = CFGA["methods"]
assert METHODS == S.M5 and CFGA["charge_classes"] == S.CLS, "config and schema dictionary disagree"
SCRIPT = Path(__file__).resolve()
RES, RAW = C.RES, C.RAW
RES.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
FAILS = []
MISSING = C.MISSING
DATASETS = [("dude", "dude_tsv"), ("pdbbind", "pdbbind_tsv")]
C.check_pins()


def work(smi):
    return C.analyze_one(smi), C.dimorphite_set_charges(smi)


def cls_of(q, smi):
    if q is None:
        return None
    if smi:
        m = C.parse_any(smi)
        if m is not None and Chem.GetFormalCharge(m) == q:
            return C.state_class(smi)
    return C.charge_class(q, max(q, 0), max(-q, 0))


def build(ds, key):
    rows = list(csv.DictReader(open(C.PATHS[key], newline="", encoding="utf-8"), delimiter="\t"))
    smis = [r["parent_smiles"] for r in rows]
    t0 = time.time()
    with mp.get_context("fork").Pool(ST["threads"]) as pool:
        res = pool.map(work, smis, chunksize=4)
    ob = C.openbabel_batch(smis)
    ml = C.run_ml(smis, ds)
    C.log(ds, "methods computed in %.0fs" % (time.time() - t0))
    out, chk = [], dict(n=len(rows), ob_recovered_unsanitised=0, ob_divalent_O_anion=0, analyze_failed=0, dom_differs_from_stored=0, single_differs_from_stored=0, ml_failed=0, ob_missing=0,
                        parent_with_formal_charge=0)
    for i, r in enumerate(rows):
        a, ddset = res[i]
        q_naive, q_dom = int(r["single_charge"]), int(r["dom_charge"])
        if a is None:
            chk["analyze_failed"] += 1
        else:
            chk["dom_differs_from_stored"] += a["dom_charge"] != q_dom
            chk["single_differs_from_stored"] += a["single_charge"] != q_naive
        m0 = Chem.MolFromSmiles(smis[i])
        chk["parent_with_formal_charge"] += bool(m0 is not None and Chem.GetFormalCharge(m0) != 0)
        single_smi = r.get("single_smiles") or (a["single_smiles"] if a and a["single_charge"] == q_naive else None)
        dom_smi = r.get("dom_smiles") or (a["dom_smiles"] if a and a["dom_charge"] == q_dom else None)
        obq = ob.get(i)
        chk["ob_missing"] += obq is None
        chk["ob_recovered_unsanitised"] += bool(obq and not obq[2])
        chk["ob_divalent_O_anion"] += bool(obq and not obq[2] and C.has_divalent_o_anion(obq[1]))
        mls = C.ml_state(ml.get(smis[i]))
        chk["ml_failed"] += mls is None
        q_pop = a["pop_charge"] if a else None
        rec = dict(dataset=ds, compound_id=r["compound_id"], parent_smiles=smis[i], q_naive=q_naive, q_dimorphite_dom=q_dom,
                   q_openbabel=obq[0] if obq and obq[2] else None, q_openbabel_raw=obq[0] if obq else None, ob_parse_ok=obq[2] if obq else None, q_enum_popmax=q_pop, q_ml_pka=mls[0] if mls else None,
                   class_naive=cls_of(q_naive, single_smi), class_dimorphite_dom=cls_of(q_dom, dom_smi),
                   class_openbabel=cls_of(obq[0], obq[1]) if obq and obq[2] else None, class_enum_popmax=cls_of(q_pop, a["pop_smiles"]) if a else None,
                   class_ml_pka=C.ml_class(ml[smis[i]]["q0"], mls[1], mls[2]) if mls else None,
                   dd_state_charges="|".join(str(c) for c in sorted(ddset)) if ddset else None,
                   dd_plausible_charged=any(c != 0 for c in ddset) if ddset else None,
                   dd_neutral_excluded=(q_naive not in ddset) if ddset else None,
                   ml_q0=ml[smis[i]].get("q0") if mls else None, ml_n_base_groups=len(ml[smis[i]]["base"]) if mls else None,
                   ml_n_acid_groups=len(ml[smis[i]]["acid"]) if mls else None)
        rec["complete_case"] = all(rec["q_" + m] is not None for m in METHODS)
        out.append(rec)
    chk["stored_mismatch_charge_sum"] = sum(int(r["mismatch_charge"]) for r in rows)
    chk["recomputed_mismatch_sum"] = sum(1 for x in out if x["q_dimorphite_dom"] != x["q_naive"])
    return out, chk


def kappa_tables(T):
    n = T.sum((-1, -2))
    po = np.trace(T, axis1=-2, axis2=-1) / n
    pe = (T.sum(-1) * T.sum(-2)).sum(-1) / n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return (po - pe) / (1 - pe)


def encode(qa, qb):
    u, inv = np.unique(np.concatenate([qa, qb]), return_inverse=True)
    return inv[:len(qa)], inv[len(qa):], len(u)


def boot_pair(blocks, rng, B, chunk=1000):
    n = len(blocks[0][0])
    onehots = []
    for ca, cb, K in blocks:
        oh = np.zeros((n, K * K), dtype=np.float32); oh[np.arange(n), ca * K + cb] = 1; onehots.append(oh)
    outs = [[] for _ in blocks]
    p = np.full(n, 1.0 / n)
    for s in range(0, B, chunk):
        w = rng.multinomial(n, p, size=min(chunk, B - s)).astype(np.float32)
        for j, (ca, cb, K) in enumerate(blocks):
            outs[j].append(kappa_tables((w @ onehots[j]).astype(np.float64).reshape(-1, K, K)))
    return [np.concatenate(o) for o in outs]


def pct(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return (float(np.percentile(x[ok], 2.5)), float(np.percentile(x[ok], 97.5)), int((~ok).sum())) if ok.any() else (float("nan"), float("nan"), len(x))


def fleiss_point(Tm):
    n, m = Tm.shape[0], Tm.sum(1)[0]
    Pi = ((Tm ** 2).sum(1) - m) / (m * (m - 1))
    pj = Tm.sum(0) / (n * m)
    Pe = (pj ** 2).sum()
    return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def fleiss_boot(Tm, rng, B, chunk=1000):
    n, m = Tm.shape[0], int(Tm.sum(1)[0])
    Pi = ((Tm ** 2).sum(1) - m) / (m * (m - 1))
    p = np.full(n, 1.0 / n); out = []
    for s in range(0, B, chunk):
        w = rng.multinomial(n, p, size=min(chunk, B - s)).astype(np.float64)
        pj = (w @ Tm) / (n * m)
        Pe = (pj ** 2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(((w @ Pi) / n - Pe) / (1 - Pe))
    return np.concatenate(out)


def tabulate(mat, cats):
    return np.stack([(mat == c).sum(1) for c in cats], axis=1).astype(np.float64)


def main():
    t_start = time.time()
    protonation = json.load(open(C.PATHS["protonation_tools_json"]))
    for ds in ("dude", "pdbbind"):
        for k, v in CFGA["original_numbers_reproduced"][ds].items():
            if protonation[ds][k] != v:
                FAILS.append("config original number %s.%s=%s != baseline file %s" % (ds, k, v, protonation[ds][k]))
    L, CHK = {}, {}
    for ds, key in DATASETS:
        L[ds], CHK[ds] = build(ds, key)
    allrows = [x for ds, _ in DATASETS for x in L[ds]]
    pl_path = RES / "charge_per_ligand.csv"
    C.write_csv(pl_path, S.PER_LIGAND, allrows)

    pair_rows, conf_files, fleiss_json = [], [], {}
    seed = ST["seed"]; B = ST["n_boot"]
    for di, (ds, _) in enumerate(DATASETS):
        Q = {m: np.array([MISSING if x["q_" + m] is None else x["q_" + m] for x in L[ds]]) for m in METHODS}
        K6 = {m: np.array([-1 if x["class_" + m] is None else S.CLS.index(x["class_" + m]) for x in L[ds]]) for m in METHODS}
        for pi, (ma, mb) in enumerate(combinations(METHODS, 2)):
            mask = (Q[ma] != MISSING) & (Q[mb] != MISSING)
            n = int(mask.sum()); k = int((Q[ma][mask] == Q[mb][mask]).sum())
            lo, hi = C.wilson(k, n)
            ca, cb, Kc = encode(Q[ma][mask], Q[mb][mask])
            ta = np.zeros((Kc, Kc)); np.add.at(ta, (ca, cb), 1)
            ka = float(kappa_tables(ta))
            ca6, cb6 = K6[ma][mask], K6[mb][mask]
            t6 = np.zeros((6, 6)); np.add.at(t6, (ca6, cb6), 1)
            k6 = float(kappa_tables(t6))
            rng = np.random.default_rng(seed + 1000 * di + pi)
            bc, b6 = boot_pair([(ca, cb, Kc), (ca6, cb6, 6)], rng, B)
            (l1, h1, u1), (l6, h6, u6) = pct(bc), pct(b6)
            pair_rows.append(dict(dataset=ds, method_a=ma, method_b=mb, n=n, n_agree=k, agree_rate=k / n, agree_lo=lo, agree_hi=hi, kappa=ka,
                                  kappa_lo=l1, kappa_hi=h1, kappa_class=k6, kappa_class_lo=l6, kappa_class_hi=h6))
            if u1 or u6:
                FAILS.append("%s %s/%s: %d + %d bootstrap kappas undefined" % (ds, ma, mb, u1, u6))
            crow = []
            for ci, c in enumerate(S.CLS):
                crow.append(dict([("class_a", c)] + [("b_" + S.CLS_TAG[d], int(((K6[ma] == ci) & (K6[mb] == di2)).sum())) for di2, d in enumerate(S.CLS)]))
            fn = RES / ("charge_confusion_%s_%s_vs_%s.csv" % (ds, ma, mb))
            C.write_csv(fn, S.CONFUSION, crow); conf_files.append(fn)
        cc = np.array([x["complete_case"] for x in L[ds]])
        fres = dict(n_total=len(cc), n_complete=int(cc.sum()), sets={})
        for si, (sname, members) in enumerate(CFGA["fleiss_sets"].items()):
            mat = np.stack([Q[m][cc] for m in members], axis=1)
            cats = sorted(set(mat.ravel().tolist()))
            Tm = tabulate(mat, cats)
            kap = fleiss_point(Tm)
            rng = np.random.default_rng(seed + 5000 + 100 * di + si)
            lo_, hi_, u_ = pct(fleiss_boot(Tm, rng, B))
            mat6 = np.stack([K6[m][cc] for m in members], axis=1)
            T6 = tabulate(mat6, list(range(6)))
            k6 = fleiss_point(T6)
            lo6, hi6, u6 = pct(fleiss_boot(T6, np.random.default_rng(seed + 7000 + 100 * di + si), B))
            if u_ or u6:
                FAILS.append("%s fleiss %s: undefined bootstrap kappas %d/%d" % (ds, sname, u_, u6))
            fres["sets"][sname] = dict(methods=members, categories_charge=cats, kappa_charge=kap, kappa_charge_lo=lo_, kappa_charge_hi=hi_,
                                       kappa_class=k6, kappa_class_lo=lo6, kappa_class_hi=hi6,
                                       all_methods_identical_fraction=float((mat == mat[:, [0]]).all(1).mean()),
                                       all_methods_identical_k=int((mat == mat[:, [0]]).all(1).sum()))
        fleiss_json[ds] = fres
    pair_path = RES / "agreement_pairwise.csv"
    C.write_csv(pair_path, S.PAIRWISE, pair_rows)

    cnt_rows = []
    for ds, _ in DATASETS:
        for m in METHODS:
            vals = [x["class_" + m] for x in L[ds] if x["class_" + m] is not None]
            for c in S.CLS:
                kk = sum(1 for v in vals if v == c)
                cnt_rows.append(dict(dataset=ds, method=m, charge_class=c, count=kk, fraction=kk / len(vals) if vals else None))
    cnt_path = RES / "method_class_counts.csv"
    C.write_csv(cnt_path, S.CLASS_COUNTS, cnt_rows)

    met_rows, metric = [], {}
    for ds, _ in DATASETS:
        X = L[ds]
        obs = [x["q_openbabel"] for x in X]
        n00 = sum(1 for x, o in zip(X, obs) if o is not None or x["dd_state_charges"] is not None)
        ob_bad = [o is not None and o != x["q_naive"] for x, o in zip(X, obs)]
        dd_abs = [x["dd_neutral_excluded"] is True for x in X]
        defs = [
            ("ob_ne_default", "OpenBabel net charge differs from the default (naive) net charge", sum(ob_bad), n00),
            ("ob_nonzero", "OpenBabel assigns a non-zero net charge (zwitterions count as neutral)", sum(1 for o in obs if o not in (None, 0)), n00),
            ("ob_unsanitised", "OpenBabel output that RDKit cannot sanitise (baseline treated these as having no OpenBabel charge; they have none here either)", sum(1 for x in X if x["ob_parse_ok"] is False), len(X)),
            ("ob_unsanitised_divalent_O_anion", "of the unsanitisable OpenBabel outputs, those with an [O-] bonded to two heavy atoms (an invalid valence: an OpenBabel artefact, so their charge is not trusted)",
             CHK[ds]["ob_divalent_O_anion"], sum(1 for x in X if x["ob_parse_ok"] is False)),
            ("ob_ne_default_recovered", "SENSITIVITY: as ob_ne_default, with the charge of unsanitisable OpenBabel outputs read from the unsanitised structure (untrusted)",
             sum(1 for x in X if x["q_openbabel_raw"] is not None and x["q_openbabel_raw"] != x["q_naive"]), len(X)),
            ("ob_nonzero_recovered", "SENSITIVITY: as ob_nonzero, with the charge of unsanitisable OpenBabel outputs read from the unsanitised structure (untrusted)",
             sum(1 for x in X if x["q_openbabel_raw"] not in (None, 0)), len(X)),
            ("dd_has_charged", "any Dimorphite state at pH 7.4 (default precision) has a non-zero net charge",
             sum(1 for x in X if x["dd_plausible_charged"] is True), n00),
            ("dd_default_absent", "the default (naive) net charge is not among the net charges of the Dimorphite states at pH 7.4", sum(dd_abs), n00),
            ("both_tools_differ", "OpenBabel and Dimorphite both exclude the default net charge", sum(1 for a, b in zip(ob_bad, dd_abs) if a and b), n00),
            ("enum_mismatch", "Dimorphite dominant net charge differs from the default net charge", sum(1 for x in X if x["q_dimorphite_dom"] != x["q_naive"]), len(X)),
            ("naive_nonzero", "the default state has a non-zero net charge", sum(1 for x in X if x["q_naive"] != 0), len(X)),
            ("dimorphite_dom_nonzero", "the Dimorphite dominant state has a non-zero net charge", sum(1 for x in X if x["q_dimorphite_dom"] != 0), len(X)),
            ("enum_popmax_nonzero", "the most populated ensemble member has a non-zero net charge",
             sum(1 for x in X if x["q_enum_popmax"] not in (None, 0)), sum(1 for x in X if x["q_enum_popmax"] is not None)),
            ("ml_pka_nonzero", "the MolGpKa-based state has a non-zero net charge",
             sum(1 for x in X if x["q_ml_pka"] not in (None, 0)), sum(1 for x in X if x["q_ml_pka"] is not None)),
            ("enum_popmax_ne_dom", "most populated ensemble member and Dimorphite dominant state differ in net charge",
             sum(1 for x in X if x["q_enum_popmax"] is not None and x["q_enum_popmax"] != x["q_dimorphite_dom"]), sum(1 for x in X if x["q_enum_popmax"] is not None)),
        ]
        for name, d, k, n in defs:
            lo, hi = C.wilson(k, n)
            met_rows.append(dict(dataset=ds, metric=name, definition=d, k=k, n=n, rate=k / n, wilson_lo=lo, wilson_hi=hi))
            metric[(ds, name)] = (k, n)
    met_path = RES / "agreement_set_metrics.csv"
    C.write_csv(met_path, S.SET_METRICS, met_rows)

    pw = {(r["dataset"], r["method_a"], r["method_b"]): r for r in pair_rows}
    original = CFGA["original_numbers_reproduced"]; ten_target_enum = CFGA["original_enumeration_mismatch"]
    QUOTED = {("dude", "ob_nonzero"): "45% (T0007, T0012, T0056, T0191, T0265, T0352)", ("dude", "dd_has_charged"): "77% (T0008, T0014, T0189, T0267)",
              ("dude", "dd_default_absent"): "9% (T0009, T0013, T0192, T0266)", ("dude", "enum_mismatch"): "65% (T0190; 'most-populated state under full enumeration' is in fact Dimorphite dominant vs default)",
              ("pdbbind", "ob_nonzero"): "53% (T0015)", ("pdbbind", "enum_mismatch"): "65% (T0016)"}
    map_rows, repro = [], {}
    for ds, _ in DATASETS:
        specs = [("ob_ne_default", original[ds]["openbabel_ne_default_frac"], "openbabel_ne_default_frac",
                  "raw agreement of OpenBabel with the default state (naive vs openbabel)", pw[(ds, "naive", "openbabel")]["n_agree"], pw[(ds, "naive", "openbabel")]["n"]),
                 ("ob_nonzero", original[ds]["openbabel_nonzero_frac"], "openbabel_nonzero_frac",
                  "share of ligands on which OpenBabel's state is net neutral, i.e. agrees with an all-neutral reference", None, None),
                 ("dd_has_charged", original[ds]["dimorphite_has_charged_state_frac"], "dimorphite_has_charged_state_frac",
                  "set-type metric kept as defined; complement = share whose Dimorphite state set at pH 7.4 is entirely net neutral", None, None),
                 ("dd_default_absent", original[ds]["dimorphite_default_absent_frac"], "dimorphite_default_absent_frac",
                  "set-type metric kept as defined; complement = share whose default net charge is contained in the Dimorphite state set", None, None),
                 ("enum_mismatch", ten_target_enum[ds], "mismatch_charge_frac",
                  "raw agreement of the Dimorphite dominant state with the default state (naive vs dimorphite_dom)", pw[(ds, "naive", "dimorphite_dom")]["n_agree"], pw[(ds, "naive", "dimorphite_dom")]["n"])]
        for name, refv, refname, adef, ak, an in specs:
            k, n = metric[(ds, name)]
            rep = k / n
            if ak is None:
                ak, an = n - k, n
            lo, hi = C.wilson(ak, an)
            ok = round(rep, 4) == refv
            repro[(ds, name)] = ok
            if not ok:
                FAILS.append("original number not reproduced: %s %s original=%s new=%.6f" % (ds, name, refv, rep))
            note = QUOTED.get((ds, name), "not quoted in the manuscript text (same metric family)")
            if name in ("ob_ne_default", "ob_nonzero"):
                kr, nr = metric[(ds, name + "_recovered")]
                note += "; with %d unsanitisable OpenBabel outputs recovered the value is %.4f (k=%d, n=%d)" % (metric[(ds, "ob_unsanitised")][0], kr / nr, kr, nr)
            if name == "ob_nonzero":
                note += "; the default state is net neutral for %d of %d ligands, so this complement differs slightly from naive-vs-OpenBabel agreement" % (
                    len(L[ds]) - metric[(ds, "naive_nonzero")][0], len(L[ds]))
            map_rows.append(dict(item_id="%s.%s" % (ds, name), dataset=ds, original_claim=note.split(" (")[0] if (ds, name) in QUOTED else "(not quoted)",
                                 original_metric=refname, original_value=refv, reproduced_value=rep, reproduced_ok=ok, agreement_definition=adef,
                                 agreement_value=ak / an, agreement_lo=lo, agreement_hi=hi, note=note))
    map_path = RES / "agreement_original_numbers.csv"
    C.write_csv(map_path, S.OLD_MAP, map_rows)

    ob_ver = subprocess.run([str(C.PATHS["obabel"]), "-V"], capture_output=True, text=True).stdout.strip()
    ml_env = subprocess.run([str(C.PATHS["ml_python"]), "-c", "import rdkit,torch,sys;print(sys.version.split()[0],rdkit.__version__,torch.__version__)"],
                            capture_output=True, text=True).stdout.strip()
    extra = dict(dimorphite_dl=metadata.version("dimorphite_dl"), openbabel=ob_ver, ml_env_python_rdkit_torch=ml_env,
                 molgpka_weight_acid_md5=C.P.file_md5(C.PATHS["molgpka_acid"]), molgpka_weight_base_md5=C.P.file_md5(C.PATHS["molgpka_base"]),
                 molgpka_commit="4dc835219baad18c2e705274ac54d49597703966", ml_wrapper_md5=C.P.file_md5(C.PATHS["ml_wrapper"]),
                 original_dominant_functions="scripts/charge_state/dominant_state.py (verbatim copy of dominant_state.py lines 19-276)",
                 ml_raw_outputs=[str(RAW / ("ml_sites_%s.jsonl" % ds)) for ds, _ in DATASETS])
    INPUTS = [C.PATHS[k] for k in ("dude_tsv", "pdbbind_tsv", "original_dominant_py", "dimorphite_smarts", "protonation_tools_json")]
    BLOCK = C.P.provenance_block(SCRIPT, C.ROOT / "configs/charge_state.yaml", INPUTS, seed=seed, n_boot=B, extra=extra, tools_extra=("statsmodels", "rdkit", "sklearn"))
    grains = S.ROW_GRAIN
    C.P.write_schema(pl_path, S.PER_LIGAND, grains["charge_per_ligand.csv"], "Per-ligand pH-7.4 net charge and charge class of five methods on DUD-E and PDBbind", extra=dict(provenance=BLOCK))
    C.P.write_schema(pair_path, S.PAIRWISE, grains["agreement_pairwise.csv"], "Pairwise agreement of the methods: raw agreement with Wilson CI, Cohen kappa on net charge and on six charge classes with bootstrap CI over ligands", extra=dict(provenance=BLOCK))
    C.P.write_schema(cnt_path, S.CLASS_COUNTS, grains["method_class_counts.csv"], "Charge-class composition of each method on each dataset", extra=dict(provenance=BLOCK))
    C.P.write_schema(met_path, S.SET_METRICS, grains["agreement_set_metrics.csv"], "Set-type and neutral-fraction metrics, including the baseline definitions of the manuscript's charge-state percentages", extra=dict(provenance=BLOCK))
    C.P.write_schema(map_path, S.OLD_MAP, grains["agreement_original_numbers.csv"], "Map from each original charge-state percentage to its reproduced value and its agreement-wording counterpart", extra=dict(provenance=BLOCK))
    for fn in conf_files:
        C.P.write_schema(fn, S.CONFUSION, grains["charge_confusion"], "Confusion matrix of charge classes between two methods (rows method a, columns method b)", extra=dict(provenance=BLOCK))
    fj = dict(provenance=BLOCK, datasets=fleiss_json, note="Fleiss kappa over ligands with a charge from all five methods; bootstrap = multinomial resampling of those ligands, 10,000 resamples")
    json.dump(fj, open(RES / "agreement_fleiss.json", "w"), indent=2, ensure_ascii=False)
    sens = {}
    qv = lambda x, m: x["q_openbabel_raw"] if m == "openbabel" else x["q_" + m]
    for ds, _ in DATASETS:
        sens[ds] = {}
        for ma, mb in combinations(METHODS, 2):
            if "openbabel" in (ma, mb):
                both = [x for x in L[ds] if qv(x, ma) is not None and qv(x, mb) is not None]
                kk = sum(1 for x in both if qv(x, ma) == qv(x, mb))
                sens[ds]["%s_vs_%s" % (ma, mb)] = dict(n=len(both), n_agree=kk, agree_rate=kk / len(both))
    summary = dict(provenance=BLOCK, checks=CHK, openbabel_recovered_charge_sensitivity=sens, original_numbers_reproduced={"%s.%s" % k: v for k, v in repro.items()}, fails=FAILS,
                   wall_seconds=round(time.time() - t_start, 1))
    json.dump(summary, open(RES / "agreement_summary.json", "w"), indent=2, ensure_ascii=False)
    C.log("wrote", len(conf_files) + 7, "files; fails:", FAILS or "none")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
