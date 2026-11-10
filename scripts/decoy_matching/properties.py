import csv, os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_pipeline as PL
import decoy_props as PR

ROOT, VF = PL.ROOT, PL.VF
RM = Path(os.environ.get("BASE_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
RAW = OUT_ROOT / "raw_outputs/decoy_matching"; PARTS = RAW / "props_parts"; PARTS.mkdir(parents=True, exist_ok=True)
DUDE = VF / "verification/C2_dude_vs/inputs/dude_all_targets"
AVE = RM / "data/raw/lit_pcba/AVE_unbiased"
AVE_DIR = {"aldh1": "ALDH1", "esr1_ant": "ESR1_ant", "fen1": "FEN1", "gba": "GBA", "kat2a": "KAT2A", "mapk1": "MAPK1", "mtorc1": "MTORC1", "pkm2": "PKM2", "tp53": "TP53", "vdr": "VDR"}
NPROC = int(os.environ.get("DECOY_NPROC", "44"))
CAP = int(CFG["ligand_sets"]["comparison_litpcba_inactives_per_target"]); ASEED = int(CFG["ligand_sets"]["comparison_seed"])
FIELDS = ["dataset", "target", "kind", "ligand_id", "smiles", "std_smiles", "ok", "error", "dominant_charge", "n_sites", "n_microstates_all", "n_selected", "n_microstates", "n_tautomers", "min_q", "max_q", "n_rot", "n_heavy", "mw", "seconds"]


def read_smi(p):
    out = []
    for ln in open(p, encoding="utf-8"):
        t = ln.split()
        if len(t) >= 2:
            out.append((t[1], t[0]))
    return out


def work(item):
    r = PR.full_props(item[3])
    r.update(dataset=item[0], target=item[1], kind=item[2], ligand_id=item[4], smiles=item[3])
    return r


def units():
    for t in CFG["targets"]["dude"]:
        yield ("dude", t, "active", read_smi(DUDE / t / "actives_final.ism"))
        yield ("dude", t, "decoy", read_smi(DUDE / t / "decoys_final.ism"))
    for i, t in enumerate(CFG["targets"]["litpcba"]):
        act = [x for k in ("T", "V") for x in read_smi(AVE / AVE_DIR[t] / ("active_%s.smi" % k))]
        ina = [x for k in ("T", "V") for x in read_smi(AVE / AVE_DIR[t] / ("inactive_%s.smi" % k))]
        ina = sorted(set(ina))
        rng = random.Random(ASEED + i)
        ina = rng.sample(ina, CAP) if len(ina) > CAP else ina
        yield ("litpcba", t, "active", act)
        yield ("litpcba", t, "inactive", ina)


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return format(v, ".8g")
    return str(v)


t0 = time.time()
with Pool(NPROC) as pool:
    for ds, t, kind, lig in units():
        part = PARTS / ("%s_%s_%s.csv" % (ds, t, kind))
        if part.exists():
            continue
        items = [(ds, t, kind, smi, lid) for lid, smi in lig]
        rows = pool.map(work, items, chunksize=32)
        tmp = part.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(FIELDS)
            for r in rows:
                w.writerow([fmt(r.get(k)) for k in FIELDS])
        tmp.rename(part)
        print("[%s] %s %s %s: %d ligands, %d failed, %.0fs" % (time.strftime("%H:%M:%S"), ds, t, kind, len(rows), sum(1 for r in rows if not r["ok"]), time.time() - t0), flush=True)

COLS = [("dataset", "", "dude or litpcba"), ("target", "", "target"), ("kind", "", "active / decoy / inactive"), ("ligand_id", "", "ChEMBL id / ZINC id (DUD-E) or PubChem-style id (LIT-PCBA)"), ("smiles", "", "SMILES of the source file"),
        ("std_smiles", "", "standardised ligand (rdkit_ingest rules of the pipeline)"), ("ok", "", "enumeration produced >= 1 docking-eligible microstate"), ("error", "", "exception text"),
        ("dominant_charge", "e", "formal charge of the Dimorphite dominant state at pH 7.4 (precision 0)"), ("n_sites", "", "ionizable groups by the charge-state independent SMARTS (N-C-N merged)"),
        ("n_microstates_all", "", "microstates before the selection rule"), ("n_selected", "", "microstates after the selection rule"), ("n_microstates", "", "docking-eligible (|q| <= 2) selected microstates"),
        ("n_tautomers", "", "distinct tautomer keys among the eligible microstates"), ("min_q", "e", "minimum formal charge of the eligible microstates"), ("max_q", "e", "maximum formal charge of the eligible microstates"),
        ("n_rot", "", "rotatable bonds of the standardised ligand"), ("n_heavy", "", "heavy atoms"), ("mw", "Da", "molecular weight"), ("seconds", "s", "wall time of the property calculation")]
for ds in ("dude", "litpcba"):
    out = RAW / ("ligand_properties_%s.csv" % ds)
    parts = sorted(PARTS.glob("%s_*.csv" % ds))
    with open(out, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo); w.writerow(FIELDS)
        for p in parts:
            rd = csv.reader(open(p, encoding="utf-8")); next(rd)
            for row in rd:
                w.writerow(row)
    BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [Path(PL.__file__), Path(PR.__file__)], seed=ASEED, extra=dict(n_parts=len(parts), nproc=NPROC, litpcba_inactive_cap=CAP))
    P.write_schema(out, COLS, "one row per ligand", "Per-ligand properties of the %s ligands used in the decoy-matching property comparison and docking sets" % ds, extra=dict(provenance=BLOCK))
    print("wrote", out, flush=True)
print("done in %.0fs" % (time.time() - t0))
