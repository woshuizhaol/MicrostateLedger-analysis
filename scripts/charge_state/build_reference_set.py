import csv, gzip, json, math, sys, time
import multiprocessing as mp
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C
import charge_schemas as S

A, ST = C.CFG["anchor"], C.CFG["stats"]
SCRIPT = Path(__file__).resolve()
C.check_pins()
C.RES.mkdir(parents=True, exist_ok=True); C.RAW.mkdir(parents=True, exist_ok=True)
GLO, GHI = A["gate_window"]
RLO, RHI = C.PH - A["relevant_halfwidth"], C.PH + A["relevant_halfwidth"]
PRIO = A["source_priority"]
LOG = dict(steps=[], notes=[])


class Chain:

    def __init__(self, source, unit, df):
        self.source, self.unit, self.df = source, unit, df
        self.n0 = len(df)

    def apply(self, mask, name, desc):
        mask = np.asarray(mask, dtype=bool)
        n_in, n_out = len(self.df), int(mask.sum())
        LOG["steps"].append(dict(source=self.source, unit=self.unit, step=name, description=desc, n_in=n_in, n_dropped=n_in - n_out, n_out=n_out))
        self.df = self.df[mask].copy()
        return self


def std_cache(smiles):
    out = {}
    for s in dict.fromkeys(smiles):
        out[s] = C.standardize(s)
    return out


def apply_molecule_filters(ch, smiles_col="SMILES"):
    cache = std_cache(ch.df[smiles_col].tolist())
    ch.df["_ok"] = [cache[s]["reason"] == "" for s in ch.df[smiles_col]]
    ch.apply(ch.df["_ok"], "standardise", "SMILES parses and survives largest-fragment choice + neutralisation")
    std = [cache[s] for s in ch.df[smiles_col]]
    ch.df["smiles"] = [x["smiles"] for x in std]; ch.df["ik"] = [x["inchikey"] for x in std]
    ch.df["skel"] = [x["inchikey"][:14] for x in std]; ch.df["mw"] = [x["mw"] for x in std]; ch.df["heavy"] = [x["heavy"] for x in std]
    ch.apply([x["elements_ok"] for x in std], "elements", "only H C N O F P S Cl Br I")
    std = [cache[s] for s in ch.df[smiles_col]]
    ch.apply([x["net_charge"] == A["required_net_charge"] for x in std], "net_charge_0", "net charge 0 after neutralisation (permanent cations and anions removed)")
    std = [cache[s] for s in ch.df[smiles_col]]
    ch.apply([C.druglike(x) for x in std], "druglike", "MW 150-600, >= 1 ring, HBD <= 5, HBA <= 10")
    return ch


def merge_constants(recs, source):
    consts, dropped_by = defaultdict(list), Counter()
    n_groups = n_drop = 0
    for (skel, d, st), g in recs.groupby(["skel", "direction", "step"], sort=True):
        n_groups += 1
        v = g["pka"].to_numpy(float)
        spread = float(v.max() - v.min())
        if spread > A["max_dispersion"]:
            n_drop += 1; dropped_by[skel] += 1
            continue
        consts[skel].append(dict(direction=d, step=int(st), pka=float(np.median(v)), n_records=len(g), spread=spread))
    out = {}
    for skel in set(recs["skel"]):
        cs = sorted(consts.get(skel, []), key=lambda c: (c["direction"], c["step"]))
        out[skel] = dict(consts=cs, dropped=dropped_by.get(skel, 0) > 0, ladder_gap=ladder_gap(cs))
    LOG["steps"].append(dict(source=source, unit="constants", step="merge_replicates",
                             description="records of one (molecule, direction, step) -> median; groups with max-min > %.1f dropped" % A["max_dispersion"],
                             n_in=n_groups, n_dropped=n_drop, n_out=n_groups - n_drop))
    return out


def ladder_gap(cs):
    for d in ("A", "AH"):
        s = sorted((c for c in cs if c["direction"] == d), key=lambda c: c["step"])
        steps = [c["step"] for c in s]
        if steps and steps != list(range(1, len(steps) + 1)):
            return True
        v = [c["pka"] for c in s]
        if d == "A" and any(b < a for a, b in zip(v, v[1:])):
            return True
        if d == "AH" and any(b > a for a, b in zip(v, v[1:])):
            return True
    return False


def load_iupac():
    d = pd.read_csv(C.PATHS["iupac_csv"], low_memory=False)
    d["T_num"] = pd.to_numeric(d["T"], errors="coerce")
    d["val"] = pd.to_numeric(d["pka_value"], errors="coerce")
    ch = Chain("IUPAC", "records", d)
    ch.apply(d.assessment.isin(A["iupac_assessment_keep"]), "assessment", "assessment in %s" % A["iupac_assessment_keep"])
    ch.apply(ch.df.T_num.notna(), "T_present", "temperature given and numeric")
    ch.apply(ch.df.T_num.between(A["t_min_c"], A["t_max_c"]), "T_range", "%.0f <= T <= %.0f C" % (A["t_min_c"], A["t_max_c"]))
    ch.apply(ch.df.cosolvent.isna() | (ch.df.cosolvent.astype(str).str.strip() == ""), "cosolvent", "no cosolvent / acid / base / D2O medium recorded")
    ch.apply(ch.df.pressure.isna() | ch.df.pressure.isin(A["pressure_ok"]), "pressure", "ambient pressure (or none recorded)")
    ext = ch.df.pka_type.astype(str).str.extract(A["pka_type_regex"])
    ch.df["base_type"], ch.df["step"] = ext[0].values, pd.to_numeric(ext[1], errors="coerce").fillna(1).astype(int).values
    ch.apply(ch.df.base_type.notna(), "pka_type", "pka_type is pKa<n>, pKaH<n> or pKb<n> (excited-state / site-specific types dropped)")
    want = ch.df.base_type.map({"pKaH": "AH", "pKa": "A", "pKb": "B"})
    ch.apply((want == ch.df.acidity_label) & ch.df.acidity_label.isin(A["label_keep"]), "label_consistent", "acidity_label agrees with pka_type (pKa-A, pKaH-AH, pKb-B)")
    ch.apply(ch.df.val.notna(), "value_numeric", "pka_value numeric")
    ch.df["direction"] = np.where(ch.df.base_type == "pKa", "A", "AH")
    ch.df["pka"] = np.where(ch.df.base_type == "pKb", A["pkb_to_pkah_offset"] - ch.df.val, ch.df.val)
    apply_molecule_filters(ch, "SMILES")
    return ch.df


def load_chembl():
    d = pd.read_csv(C.PATHS["chembl_tsv"], sep="\t", low_memory=False)
    ch = Chain("ChEMBL", "records", d)
    ch.apply(d.activity_comment.isin(["Acidic pKa", "Basic pKa"]), "typed", "activity_comment is 'Acidic pKa' or 'Basic pKa' (the only records with an acid/base type)")
    ch.apply(ch.df.standard_relation.fillna("").isin(A["chembl_relation_ok"]), "relation", "relation empty or '='")
    ch.df["val"] = pd.to_numeric(ch.df.standard_value, errors="coerce")
    ch.apply(ch.df.val.notna(), "value_numeric", "standard_value numeric")
    ch.apply(ch.df.canonical_smiles.notna(), "smiles_present", "structure present")
    ch.df["SMILES"] = ch.df.canonical_smiles
    ch.df["direction"] = np.where(ch.df.activity_comment == "Acidic pKa", "A", "AH")
    ch.df["step"], ch.df["pka"] = 1, ch.df.val
    apply_molecule_filters(ch, "SMILES")
    return ch.df


def read_sampl_csv(path):
    with open(path, newline=None, encoding="utf-8-sig") as f:
        return [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_sampl():
    mols = []
    r = read_sampl_csv(C.PATHS["sampl6_pka"]); h = r[0]
    ism = h.index("canonical isomeric SMILES"); iid = h.index("Molecule ID")
    cols = [i for i, c in enumerate(h) if c.endswith("mean")]
    for row in r[1:]:
        mols.append(("SAMPL6", row[iid], row[ism], [v for v in (num(row[i]) for i in cols) if v is not None]))
    smi7 = {row[0]: row[1] for row in read_sampl_csv(C.PATHS["sampl7_smiles"])[1:]}
    r = read_sampl_csv(C.PATHS["sampl7_pka"]); h = r[0]; ip = h.index("pKa")
    for row in r[1:]:
        mols.append(("SAMPL7", row[0], smi7.get(row[0]), [v for v in [num(row[ip])] if v is not None]))
    smi8 = {row[0]: row[1] for row in read_sampl_csv(C.PATHS["sampl8_smiles"])[1:]}
    for row in read_sampl_csv(C.PATHS["sampl8_pka"])[1:]:
        vals = [num(row[i]) for i in range(1, len(row), 2)]
        mols.append(("SAMPL8", row[0], smi8.get(row[0]), [v for v in vals if v is not None]))
    df = pd.DataFrame(mols, columns=["set", "id", "SMILES", "pkas"])
    ch = Chain("SAMPL", "molecules", df)
    ch.apply(ch.df.pkas.map(len) > 0, "has_pka", "at least one measured pKa (N/A rows dropped)")
    ch.apply(ch.df.SMILES.notna(), "smiles_present", "structure present")
    apply_molecule_filters(ch, "SMILES")
    keep, types = [], []
    for s, pk in zip(ch.df.smiles, ch.df.pkas):
        m = Chem.MolFromSmiles(s)
        kinds = {x[1] for x in C.independent_sites(m) if x[3] <= GHI and x[4] >= GLO}
        keep.append(len(kinds) == 1); types.append(next(iter(kinds)) if len(kinds) == 1 else None)
    ch.df["_kind"] = types
    ch.apply(keep, "type_resolved", "all independent-SMARTS sites inside the pH 2-12 window are of one type (acid or base); mixed / no site -> excluded (sampl_type_unresolved)")
    margin = C.CFG["post_freeze"]["sampl_type_check_margin"]
    ok_type = []
    for s, pk, kind in zip(ch.df.smiles, ch.df.pkas, ch.df["_kind"]):
        wins = [(x[3] - margin, x[4] + margin) for x in C.independent_sites(Chem.MolFromSmiles(s)) if x[1] == kind]
        ok_type.append(all(any(lo <= v <= hi for lo, hi in wins) for v in pk))
    ch.apply(ok_type, "type_consistent", "every constant lies within the nominal pKa window +/- %.1f of an independent site of the assigned type; otherwise excluded (sampl_type_inconsistent)" % margin)
    rows = []
    for _, r_ in ch.df.iterrows():
        d = "AH" if r_["_kind"] == "base" else "A"
        v = sorted(r_["pkas"], reverse=(d == "AH"))
        for k, x in enumerate(v, 1):
            rows.append(dict(skel=r_["skel"], smiles=r_["smiles"], ik=r_["ik"], mw=r_["mw"], heavy=r_["heavy"], direction=d, step=k, pka=x, sampl_set=r_["set"]))
    return pd.DataFrame(rows)


def hh_features(consts):
    cs = [(c["direction"], c["pka"]) for c in consts]
    pmf = C.hh_pmf(cs)
    q, pop = C.dominant_of(pmf)
    ql, _ = C.ladder_dominant(cs)
    inwin = sorted(c["pka"] for c in consts if GLO <= c["pka"] <= GHI)
    coupled = any(b - a < A["coupled_gap"] for a, b in zip(inwin, inwin[1:]))
    return q, pop, ql, coupled


def method_work(smi):
    a = C.analyze_one(smi)
    return None if a is None else (a["single_charge"], a["dom_charge"], a["pop_charge"])


def main():
    t0 = time.time()
    iu = load_iupac(); ch_ = load_chembl(); sm = load_sampl()
    C.log("sources loaded: IUPAC records", len(iu), "ChEMBL records", len(ch_), "SAMPL constants", len(sm), "(%.0fs)" % (time.time() - t0))
    ladders = {"IUPAC": merge_constants(iu, "IUPAC"), "ChEMBL": merge_constants(ch_, "ChEMBL"), "SAMPL": merge_constants(sm, "SAMPL")}
    meta = {}
    for name, df in (("IUPAC", iu), ("ChEMBL", ch_), ("SAMPL", sm)):
        for skel, g in df.groupby("skel"):
            meta.setdefault(skel, dict(smiles=g["smiles"].iloc[0], mw=float(g["mw"].iloc[0]), heavy=int(g["heavy"].iloc[0])))
    usable = {s: {k: v for k, v in L.items() if v["consts"]} for s, L in ladders.items()}
    for s, L in ladders.items():
        LOG["steps"].append(dict(source=s, unit="molecules", step="usable_ladder", description="molecules with at least one constant left after replicate merging",
                                 n_in=len(L), n_dropped=len(L) - len(usable[s]), n_out=len(usable[s])))
    skels = sorted(set().union(*[set(u) for u in usable.values()]))
    LOG["notes"].append("distinct molecules over all sources: %d; per source %s" % (len(skels), {s: len(u) for s, u in usable.items()}))
    feats = {}
    for skel in skels:
        m = Chem.MolFromSmiles(meta[skel]["smiles"]); sc = C.site_counts(m, meta[skel]["smiles"])
        feats[skel] = sc
    chosen, also = {}, {}
    for skel in skels:
        have = [s for s in PRIO if skel in usable[s]]
        def rank(s):
            L = usable[s][skel]; cs = L["consts"]
            nw = sum(1 for c in cs if GLO <= c["pka"] <= GHI)
            strict = nw >= max(feats[skel]["n_dd_strict"], feats[skel]["n_ind_strict"]) and not L["dropped"] and not L["ladder_gap"] and (nw >= 1 or not A["require_window_constant"])
            return (not strict, PRIO.index(s))
        best = min(have, key=rank)
        chosen[skel], also[skel] = best, [s for s in have if s != best]
    cnt = Counter(chosen.values())
    LOG["steps"].append(dict(source="ALL", unit="molecules", step="source_choice", description="one source per molecule (strict-gate-passing source first, then priority order)",
                             n_in=len(skels), n_dropped=0, n_out=len(skels), by_source=dict(cnt), n_with_second_source=sum(1 for v in also.values() if v)))
    leak = {n: set(gzip.open(C.ROOT / "raw_outputs/charge_state" / ("leak_keys_%s.txt.gz" % n), "rt").read().split()) for n in ("molgpka_example", "pkasolver_train", "chembl37")}
    smis = [meta[k]["smiles"] for k in skels]
    with mp.get_context("fork").Pool(ST["threads"]) as pool:
        res = pool.map(method_work, smis, chunksize=4)
    ob = C.openbabel_batch(smis)
    ml = C.run_ml(smis, "anchor")
    rows, crows = [], []
    for i, skel in enumerate(skels):
        s = chosen[skel]; L = usable[s][skel]; cs = L["consts"]; sc = feats[skel]
        nw = sum(1 for c in cs if GLO <= c["pka"] <= GHI); nr = sum(1 for c in cs if RLO <= c["pka"] <= RHI)
        q, pop, ql, coupled = hh_features(cs)
        n_pred, n_rel = max(sc["n_dd_strict"], sc["n_ind_strict"]), max(sc["n_dd_relaxed"], sc["n_ind_relaxed"])
        clean = (not L["dropped"]) and (not L["ladder_gap"]) and (nw >= 1 or not A["require_window_constant"])
        complete = clean and nw >= n_pred
        relaxed = complete or (clean and nr >= n_rel)
        conc = None
        if also[skel]:
            conc = all(hh_features(usable[o][skel]["consts"])[0] == q for o in also[skel])
        ms = res[i]; o = ob.get(i); mls = C.ml_state(ml.get(meta[skel]["smiles"]))
        rows.append(dict(
            mol_id="", smiles=meta[skel]["smiles"], inchikey=Chem.MolToInchiKey(Chem.MolFromSmiles(meta[skel]["smiles"])), ik_skeleton=skel, source=s,
            sources_also="|".join(also[skel]), sources_concordant=conc, mw=meta[skel]["mw"], heavy_atoms=meta[skel]["heavy"],
            n_sites_dd=sc["n_dd_strict"], n_sites_ind=sc["n_ind_strict"], n_sites_pred=n_pred, n_sites_rel=n_rel,
            n_pka_exp=len(cs), n_pka_exp_window=nw, n_pka_exp_relevant=nr, complete_profile=complete, complete_relaxed=relaxed,
            had_dropped_constant=L["dropped"], ladder_gap=L["ladder_gap"], exp_dominant_charge=q, exp_pop_max=pop, ambiguous=pop < A["ambiguous_below"],
            coupled_sites=coupled, exp_ladder_charge=ql, exp_ladder_agrees=(ql == q), functional_class=sc["primary_class"], classes_all="|".join(sc["classes"]),
            q_naive=ms[0] if ms else None, q_dimorphite_dom=ms[1] if ms else None, q_openbabel=o[0] if o and o[2] else None, q_enum_popmax=ms[2] if ms else None,
            q_ml_pka=mls[0] if mls else None, ml_in_molgpka_example_sdf=skel in leak["molgpka_example"], ml_in_pkasolver_train=skel in leak["pkasolver_train"],
            ml_in_chembl37=skel in leak["chembl37"]))
        for c in cs:
            crows.append(dict(mol_id="", source=s, direction=c["direction"], step=c["step"], pka=c["pka"], n_records=c["n_records"], spread=c["spread"],
                              in_window=GLO <= c["pka"] <= GHI, _skel=skel))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["ik_skeleton"])
    idmap = {}
    for n, i in enumerate(order, 1):
        rows[i]["mol_id"] = idmap[rows[i]["ik_skeleton"]] = "A%04d" % n
    rows = [rows[i] for i in order]
    for c in crows:
        c["mol_id"] = idmap[c.pop("_skel")]
    crows.sort(key=lambda c: (c["mol_id"], c["direction"], c["step"]))
    p_set, p_con = C.RES / "anchor_set.csv", C.RES / "anchor_constants.csv"
    C.write_csv(p_set, S.ANCHOR, rows); C.write_csv(p_con, S.CONSTANTS, crows)
    nrow = len(rows)
    final = dict(n_molecules=nrow, by_source=dict(Counter(r["source"] for r in rows)), n_complete_profile=sum(r["complete_profile"] for r in rows),
                 n_complete_relaxed=sum(r["complete_relaxed"] for r in rows), n_sensitivity_not_complete=sum(not r["complete_profile"] for r in rows),
                 n_ambiguous=sum(r["ambiguous"] for r in rows), n_coupled=sum(r["coupled_sites"] for r in rows),
                 n_ladder_disagrees=sum(not r["exp_ladder_agrees"] for r in rows), n_ladder_gap=sum(r["ladder_gap"] for r in rows),
                 n_had_dropped_constant=sum(r["had_dropped_constant"] for r in rows),
                 n_methods_failed=dict((m, sum(r["q_" + m] is None for r in rows)) for m in ("naive", "dimorphite_dom", "openbabel", "enum_popmax", "ml_pka")),
                 n_multi_source=sum(1 for r in rows if r["sources_also"]), n_multi_source_concordant=sum(1 for r in rows if r["sources_concordant"] is True))
    bad = [x for x in LOG["steps"] if x["n_in"] != x["n_dropped"] + x["n_out"]]
    assert not bad, bad
    ob_ver = __import__("subprocess").run([str(C.PATHS["obabel"]), "-V"], capture_output=True, text=True).stdout.strip()
    extra = dict(dimorphite_dl=metadata.version("dimorphite_dl"), openbabel=ob_ver, molgpka_commit="4dc835219baad18c2e705274ac54d49597703966",
                 molgpka_weight_acid_md5=C.P.file_md5(C.PATHS["molgpka_acid"]), molgpka_weight_base_md5=C.P.file_md5(C.PATHS["molgpka_base"]),
                 leak_key_files=sorted(str(p) for p in (C.ROOT / "raw_outputs/charge_state").glob("leak_keys_*.txt.gz")))
    INPUTS = [C.PATHS[k] for k in ("iupac_csv", "sampl6_pka", "sampl7_smiles", "sampl7_pka", "sampl8_smiles", "sampl8_pka", "chembl_tsv", "dimorphite_smarts")]
    BLOCK = C.P.provenance_block(SCRIPT, C.ROOT / "configs/charge_state.yaml", INPUTS, seed=ST["seed"], extra=extra, tools_extra=("rdkit",))
    C.P.write_schema(p_set, S.ANCHOR, S.ROW_GRAIN["anchor_set.csv"], "Experimental anchor: drug-like neutral molecules with a measured pKa ladder, completeness gates, experimental pH-7.4 dominant charge and method charges", extra=dict(provenance=BLOCK))
    C.P.write_schema(p_con, S.CONSTANTS, S.ROW_GRAIN["anchor_constants.csv"], "Macroscopic pKa constants of the anchor molecules after merging replicate measurements", extra=dict(provenance=BLOCK))
    json.dump(dict(provenance=BLOCK, steps=LOG["steps"], notes=LOG["notes"], final=final, wall_seconds=round(time.time() - t0, 1)),
              open(C.RES / "anchor_build_log.json", "w"), indent=2, ensure_ascii=False)
    C.log("anchor molecules", nrow, "final", final)


if __name__ == "__main__":
    main()
