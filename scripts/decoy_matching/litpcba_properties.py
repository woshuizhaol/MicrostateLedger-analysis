import csv, os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_pipeline as PL
import decoy_props as PR

ROOT = PL.ROOT
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RAW = OUT_ROOT / "raw_outputs/decoy_matching"
NPROC = int(os.environ.get("DECOY_NPROC", "2"))
FIELDS = ["dataset", "target", "kind", "ligand_id", "smiles", "std_smiles", "ok", "error", "dominant_charge", "n_sites", "n_microstates_all", "n_selected", "n_microstates", "n_tautomers", "min_q", "max_q", "n_rot", "n_heavy", "mw", "seconds"]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return format(v, ".8g")
    return str(v)


def work(item):
    r = PR.full_props(item[3])
    r.update(dataset=item[0], target=item[1], kind=item[2], ligand_id=item[4], smiles=item[3])
    return r


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    t0 = time.time()
    have = {(r["target"], r["smiles"]) for r in read_rows(RAW / "ligand_properties_litpcba.csv")}
    items, seen, n_q = [], set(), 0
    for r in read_rows(RAW / "queue/ligand_sets_sliced.csv"):
        if r["set"] != "litpcba":
            continue
        n_q += 1
        tgt = r["target"][len("litpcba_"):]
        key = (tgt, r["smiles"])
        if key in have or key in seen:
            continue
        seen.add(key)
        items.append(("litpcba", tgt, "active" if r["role"] == "active" else "inactive", r["smiles"], r["ligand_id"]))
    items.sort(key=lambda x: (x[1], x[2], x[4]))
    print("[%s] LIT-PCBA queue rows %d | without a property row %d | by target %s" % (time.strftime("%H:%M:%S"), n_q, len(items), {t: sum(1 for i in items if i[1] == t) for t in sorted({i[1] for i in items})}), flush=True)
    with Pool(NPROC) as pool:
        rows = pool.map(work, items, chunksize=4)
    out = RAW / "ligand_properties_litpcba_queue_extra.csv"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(FIELDS)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in FIELDS])
    tmp.rename(out)
    now = {(r["target"], r["smiles"]) for r in read_rows(out)} | have
    still = sum(1 for r in read_rows(RAW / "queue/ligand_sets_sliced.csv") if r["set"] == "litpcba" and (r["target"][len("litpcba_"):], r["smiles"]) not in now)
    desc = {"dataset": "litpcba", "target": "target (short name, as in ligand_properties_litpcba.csv)", "kind": "active / inactive (queue role)", "ligand_id": "PubChem-style id", "smiles": "SMILES of the source file",
            "std_smiles": "standardised ligand", "ok": "enumeration produced >= 1 docking-eligible microstate", "error": "exception text"}
    COLS = [(c, "", desc.get(c, "same definition as in ligand_properties_litpcba.csv")) for c in FIELDS]
    BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RAW / "ligand_properties_litpcba.csv", RAW / "queue/ligand_sets_sliced.csv"], seed=None, extra=dict(n_queue_rows=n_q, n_missing=len(items)))
    P.write_schema(out, COLS, "one row per ligand", "Properties of the LIT-PCBA queue ligands that are not in the 10,000-inactive property sample (used by the gain strata only, not by the property comparison)", extra=dict(provenance=BLOCK))
    print("[%s] wrote %s: %d ligands, %d with ok=True, %d failed | queue ligands still without a property row: %d | %.0f s" % (time.strftime("%H:%M:%S"), out, len(rows), sum(1 for r in rows if r["ok"]), sum(1 for r in rows if not r["ok"]), still, time.time() - t0), flush=True)
    sys.exit(1 if still else 0)
