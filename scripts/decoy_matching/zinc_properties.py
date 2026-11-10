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
NPROC = int(os.environ.get("DECOY_NPROC", "6"))
FIELDS = ["set", "target", "kind", "ligand_id", "smiles", "std_smiles", "ok", "error", "dominant_charge", "n_sites", "n_microstates_all", "n_selected", "n_microstates", "n_tautomers", "min_q", "max_q", "n_rot", "n_heavy", "mw", "seconds"]
rows = [r for r in csv.DictReader(open(RAW / "queue/ligand_sets.csv", encoding="utf-8")) if r["set"].startswith("zinc_")]
uniq = sorted({r["smiles"] for r in rows})
t0 = time.time()
with Pool(NPROC) as pool:
    res = dict(zip(uniq, pool.map(PR.full_props, uniq, chunksize=16)))
print("computed %d unique molecules in %.0f s" % (len(uniq), time.time() - t0), flush=True)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return format(v, ".8g")
    return str(v)


out = RAW / "ligand_properties_zinc_drawn.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(FIELDS)
    for r in rows:
        p = dict(res[r["smiles"]]); p.update({"set": r["set"], "target": r["target"], "kind": "negative", "ligand_id": r["ligand_id"], "smiles": r["smiles"]})
        w.writerow([fmt(p.get(k)) for k in FIELDS])
COLS = [("set", "", "zinc_unmatched / zinc_charge_mismatch"), ("target", "", "DUD-E target the molecule was drawn for"), ("kind", "", "negative"), ("ligand_id", "", "ZINC id"), ("smiles", "", "SMILES of the ZINC pool"),
        ("std_smiles", "", "standardised ligand"), ("ok", "", "enumeration produced >= 1 docking-eligible microstate"), ("error", "", "exception text"), ("dominant_charge", "e", "formal charge of the Dimorphite dominant state at pH 7.4"),
        ("n_sites", "", "ionizable groups (charge-state independent SMARTS, N-C-N merged)"), ("n_microstates_all", "", "microstates before the selection rule"), ("n_selected", "", "after the selection rule"),
        ("n_microstates", "", "docking-eligible (|q| <= 2) microstates"), ("n_tautomers", "", "distinct tautomer keys among them"), ("min_q", "e", "minimum formal charge of the eligible microstates"), ("max_q", "e", "maximum"),
        ("n_rot", "", "rotatable bonds"), ("n_heavy", "", "heavy atoms"), ("mw", "Da", "molecular weight"), ("seconds", "s", "wall time")]
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [RAW / "queue/ligand_sets.csv", Path(PL.__file__), Path(PR.__file__)], seed=None, extra=dict(n_unique=len(uniq), nproc=NPROC))
P.write_schema(out, COLS, "one row per (zinc set, target, molecule)", "Per-ligand properties of the ZINC molecules drawn for the two ZINC negative sets", extra=dict(provenance=BLOCK))
print("wrote", out, len(rows), "rows; failed:", sum(1 for r in res.values() if not r["ok"]), flush=True)
