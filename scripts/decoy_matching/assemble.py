import csv, hashlib, json, math, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RAW, RES = OUT_ROOT / "raw_outputs/decoy_matching", OUT_ROOT / "results/decoy_matching"; RES.mkdir(parents=True, exist_ok=True)
MAXR = int(sys.argv[sys.argv.index("--max-round") + 1]) if "--max-round" in sys.argv else None
MAXS = int(sys.argv[sys.argv.index("--max-slice") + 1]) if "--max-slice" in sys.argv else None
SETS = ["dude_full", "litpcba", "zinc_unmatched", "zinc_charge_mismatch"]


def uid_of(target, smiles):
    return hashlib.md5(("%s|%s" % (target, smiles)).encode()).hexdigest()[:12]


def klass(q):
    if q is None or (isinstance(q, float) and math.isnan(q)):
        return None
    q = int(q)
    return "<=-1" if q <= -1 else ("0" if q == 0 else ("+1" if q == 1 else ">=+2"))


def sites_class(n):
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return None
    return "0" if int(n) == 0 else ("1" if int(n) == 1 else ">=2")


t0 = time.time()
q = pd.read_csv(RAW / "queue/ligand_sets.csv", dtype=str, keep_default_na=False)
q["round"] = q["round"].astype(int)
q["uid"] = [uid_of(t, s) for t, s in zip(q.target, q.smiles)]
rec = {}
for f in sorted((RAW / "dock").glob("*.jsonl")):
    if f.name == "errors.jsonl":
        continue
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        ms = r["microstates"]
        rec[r["uid"]] = dict(status=r["status"], n_eligible=r["n_eligible"], std_smiles=r.get("std_smiles", ""), micro=[(m["order"], m["microstate_id"], m["formal_charge"], m.get("score") if m["status"] == "ok" else None) for m in ms],
                             cpu=r.get("cpu_seconds"))
pd_ = pd.read_csv(RAW / "ligand_properties_dude.csv", dtype={"ligand_id": str, "smiles": str}, keep_default_na=False, na_values=[""])
pl_ = pd.read_csv(RAW / "ligand_properties_litpcba.csv", dtype={"ligand_id": str, "smiles": str}, keep_default_na=False, na_values=[""])
plx_f = RAW / "ligand_properties_litpcba_queue_extra.csv"
if plx_f.exists():
    pl_ = pd.concat([pl_, pd.read_csv(plx_f, dtype={"ligand_id": str, "smiles": str}, keep_default_na=False, na_values=[""])], ignore_index=True)
pz_f = RAW / "ligand_properties_zinc_drawn.csv"
pz_ = pd.read_csv(pz_f, dtype={"ligand_id": str, "smiles": str}, keep_default_na=False, na_values=[""]) if pz_f.exists() else pd.DataFrame(columns=pd_.columns)
props = {}
for df, key in ((pd_, lambda r: ("dude", r.target, r.smiles)), (pl_, lambda r: ("litpcba", "litpcba_" + r.target, r.smiles)), (pz_, lambda r: (r.set, r.target, r.smiles) if "set" in pz_.columns else None)):
    for r in df.itertuples():
        k = key(r)
        if k:
            props[k] = (r.dominant_charge, r.n_sites, r.n_heavy)


def prop_of(set_name, target, role, smiles):
    if set_name == "dude_full":
        return props.get(("dude", target, smiles))
    if set_name == "litpcba":
        return props.get(("litpcba", target, smiles))
    if role == "active":
        return props.get(("dude", target, smiles))
    return props.get((set_name, target, smiles))


act = q[(q.set == "dude_full") & (q.role == "active")]
rows = []
for r in q.itertuples():
    rows.append((r.set, r.target, r.role, r.round, r.ligand_id, r.smiles, r.uid))
for r in act.itertuples():
    for s in ("zinc_unmatched", "zinc_charge_mismatch"):
        rows.append((s, r.target, "active", 0, r.ligand_id, r.smiles, r.uid))
full = pd.DataFrame(rows, columns=["set", "target", "role", "round", "ligand_id", "smiles", "uid"])
SLICED = RAW / "queue/ligand_sets_sliced.csv"
qs = pd.read_csv(SLICED, dtype=str, keep_default_na=False)
qs["uid"] = [uid_of(t, s) for t, s in zip(qs.target, qs.smiles)]
assert len(qs) == len(q) and set(zip(qs.set, qs.uid)) == set(zip(q.set, q.uid)), "the sliced queue is not a permutation of the queue"
slice_of = {(a, b): int(c) for a, b, c in zip(qs.set, qs.uid, qs["slice"])}
full["slice"] = [slice_of[("dude_full" if (r.role == "active" and r.set in ("zinc_unmatched", "zinc_charge_mismatch")) else r.set, r.uid)] for r in full.itertuples()]
N_SLICES = int(full["slice"].max())
full = full.assign(done=full.uid.isin(rec))
complete = {}
for (s, t), g in full.groupby(["set", "target"]):
    by = g.groupby("slice").done.all()
    k = 0
    for sl in range(1, N_SLICES + 1):
        if sl in by.index and not bool(by[sl]):
            break
        k = sl
    complete[(s, t)] = k
max_slice = MAXS if MAXS is not None else min(complete.values())
max_round = MAXR if MAXR is not None else 3
out_rows = []
for r in full.itertuples():
    d = rec.get(r.uid)
    base = dict(set=r.set, target=r.target, role=r.role, round=r.round, slice=r.slice, ligand_id=r.ligand_id, uid=r.uid, smiles=r.smiles)
    p = prop_of(r.set, r.target, r.role, r.smiles)
    base.update(dominant_charge=p[0] if p else None, n_sites=p[1] if p else None, n_heavy=p[2] if p else None)
    base["docked_record"] = d is not None
    if d is None:
        out_rows.append(dict(base, n_eligible=None, n_docked=None, default_docked=None, score_default=None, score_min=None, score_min_nd=None, score_first4_min=None, gain=None, gain_first4=None, swing=None, in_analysis=False, in_analysis_incl_single=False, in_prefix=bool(r.slice <= max_slice and r.round <= max_round)))
        continue
    ms = d["micro"]
    sc = [(o, s) for (o, mid, q_, s) in ms if s is not None]
    default = next((s for (o, mid, q_, s) in ms if o == 0 and s is not None), None)
    nd = [s for (o, s) in sc if o != 0]
    first4 = [s for (o, s) in sc if o < 4]
    ok = default is not None and len(nd) >= 1 and r.slice <= max_slice and r.round <= max_round
    ok2 = default is not None and r.slice <= max_slice and r.round <= max_round
    smin = min(s for _, s in sc) if sc else None
    out_rows.append(dict(base, n_eligible=d["n_eligible"], n_docked=len(sc), default_docked=default is not None, score_default=default, score_min=smin, score_min_nd=min(nd) if nd else None,
                         score_first4_min=min(first4) if first4 else None, gain=(default - smin) if default is not None else None, gain_first4=(default - min(first4)) if default is not None and first4 else None,
                         swing=(max(s for _, s in sc) - smin) if sc else None, in_analysis=bool(ok), in_analysis_incl_single=bool(ok2), in_prefix=bool(r.slice <= max_slice and r.round <= max_round)))
tab = pd.DataFrame(out_rows)
tab["charge_class"] = [klass(x) for x in tab.dominant_charge]
tab["sites_class"] = [sites_class(x) for x in tab.n_sites]
COLS = [("set", "", "dude_full / litpcba / zinc_unmatched / zinc_charge_mismatch"), ("target", "", "receptor target"), ("role", "", "active / negative"), ("round", "", "0 = active, k = k-th negative of the active"), ("slice", "", "Amendment 1 slice of the ligand (1..12; the ZINC-set actives carry the slice of their dude_full row)"),
        ("ligand_id", "", "database id"), ("uid", "", "md5(target|smiles)[:12], key of the docking records"), ("smiles", "", "SMILES from the source file"), ("dominant_charge", "e", "Dimorphite dominant state charge at pH 7.4"),
        ("n_sites", "", "ionizable groups (charge-state SMARTS)"), ("n_heavy", "", "heavy atoms"), ("docked_record", "", "a docking record exists"), ("n_eligible", "", "docking-eligible microstates"),
        ("n_docked", "", "microstates with a successful docking"), ("default_docked", "", "the default (first eligible) microstate was docked"), ("score_default", "kcal/mol", "Vina top-1 score of the default microstate"),
        ("score_min", "kcal/mol", "best score over the docked microstates (oracle)"), ("score_min_nd", "kcal/mol", "best score over the docked non-default microstates"), ("score_first4_min", "kcal/mol", "best of the first 4 eligible microstates"),
        ("gain", "kcal/mol", "score_default - score_min (>= 0)"), ("gain_first4", "kcal/mol", "score_default - score_first4_min"), ("swing", "kcal/mol", "max - min over docked microstates"),
        ("in_analysis", "", "default docked, >= 1 other microstate docked, slice <= max_slice, round <= max_round"),
        ("in_analysis_incl_single", "", "sensitivity set: as in_analysis but a ligand with only the default microstate docked is kept (gain = 0 by definition)"),
        ("in_prefix", "", "the ligand belongs to the analysis unit of Amendment 1 (slice <= max_slice and round <= max_round), whether or not it was docked; the virtual-screening metrics use only these"), ("charge_class", "", "<=-1 / 0 / +1 / >=+2"), ("sites_class", "", "0 / 1 / >=2")]
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RAW / "queue/ligand_sets.csv", SLICED, RAW / "ligand_properties_dude.csv", RAW / "ligand_properties_litpcba.csv"] + ([plx_f] if plx_f.exists() else []), seed=None, extra=dict(max_round=int(max_round), max_slice=int(max_slice), n_records=len(rec)))
summ = dict(provenance=BLOCK, max_round=int(max_round), max_slice=int(max_slice), n_records=len(rec), complete_slice_by_group={"%s/%s" % k: v for k, v in complete.items()}, sets={})
for s in SETS:
    sub = tab[tab.set == s].copy()
    sub = sub.sort_values(["round", "target", "role", "ligand_id", "smiles"]).reset_index(drop=True)
    path = RES / ("%s_per_ligand.csv" % s)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([c[0] for c in COLS])
        for r in sub.itertuples(index=False):
            d = r._asdict()
            w.writerow(["" if d[c[0]] is None or (isinstance(d[c[0]], float) and not np.isfinite(d[c[0]])) else (format(d[c[0]], ".8g") if isinstance(d[c[0]], float) else d[c[0]]) for c in COLS])
    P.write_schema(path, COLS, "one row per ligand of the set", "Per-ligand docking summary and gain, %s" % s, extra=dict(provenance=BLOCK))
    a, n = sub[sub.role == "active"], sub[sub.role == "negative"]
    ia, in_ = a[a.in_analysis], n[n.in_analysis]
    ia2, in2 = a[a.in_analysis_incl_single], n[n.in_analysis_incl_single]
    summ["sets"][s] = dict(n_active=len(a), n_negative=len(n), n_active_in=len(ia), n_negative_in=len(in_), n_active_in_incl_single=len(ia2), n_negative_in_incl_single=len(in2), excluded_active_rate=1 - len(ia) / max(1, len(a)), excluded_negative_rate=1 - len(in_) / max(1, len(n)),
                           active_no_record=int((~a.docked_record).sum()), negative_no_record=int((~n.docked_record).sum()),
                           excl_single_microstate_active=int(((a.n_docked.fillna(0) < 2) & a.docked_record).sum()), excl_single_microstate_negative=int(((n.n_docked.fillna(0) < 2) & n.docked_record).sum()),
                           excl_default_failed_active=int(((a.default_docked == False) & a.docked_record).sum()), excl_default_failed_negative=int(((n.default_docked == False) & n.docked_record).sum()))
json.dump(summ, open(RES / "assemble_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in summ.items() if k != "provenance"}, indent=1)[:3000])
