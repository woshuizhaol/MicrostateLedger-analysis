import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pose_validity"))
import reference_criteria as G
import posebusters_runner as PB

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
OUT = Path(os.environ.get("CRITERIA_OUT_ROOT", str(ROOT)))
RAW = OUT / "raw_outputs/pose_criteria/redock"; CHK = OUT / "raw_outputs/pose_criteria/checks"; CHK.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get("CRITERIA_CHK_NPROC", "4")); LIMIT = int(os.environ.get("CRITERIA_CHK_LIMIT", "0"))


def items_of(pdbid):
    out = []
    for ln in open(RAW / (pdbid + ".jsonl"), encoding="utf-8"):
        r = json.loads(ln)
        for s, v in r["seeds"].items():
            if v.get("status") == "ok":
                out.append(dict(pdbid=pdbid, microstate_id=r["microstate_id"], seed=int(s), vina_score=float(v["score"]), conf_rank=int(v["conf_rank"]), path=str(RAW / pdbid / ("%s_%s.sdf" % (r["microstate_id"], s)))))
    return out


def work(pdbid):
    t0 = time.time()
    items = items_of(pdbid)
    out = CHK / (pdbid + ".parquet")
    if out.exists() and len(pd.read_parquet(out, columns=["seed"])) == len(items):
        return pdbid, 0, 0.0, "kept"
    df = pd.DataFrame(items)
    if df.empty:
        return pdbid, 0, 0.0, "no_pose"
    mols = [G.first_mol(p) if Path(p).exists() else None for p in df.path]
    bust, info = PB.bust_mols_guarded(mols, G.CORE / pdbid / ("%s_protein.pdb" % pdbid))
    v = PB.verdicts(bust)
    crystal = G.crystal_ligand(pdbid)
    try:
        cbits = G.fingerprint_bits(pdbid, crystal)
        cerr = ""
    except Exception as exc:
        cbits, cerr = None, "crystal_fingerprint_error:" + str(exc)[:60]
    rm, rr, pt, pr_, nb = [], [], [], [], []
    for m in mols:
        if m is None:
            rm.append(np.nan); rr.append("pose_unreadable"); pt.append(np.nan); pr_.append("pose_unreadable"); nb.append(-1); continue
        if crystal is None:
            r_, why = np.nan, "crystal_unreadable"
        else:
            r_, why = G.rmsd_sym(crystal, m)
        rm.append(r_); rr.append(why)
        if cbits is None:
            pt.append(np.nan); pr_.append(cerr or "crystal_unreadable"); nb.append(-1); continue
        try:
            pb_ = G.fingerprint_bits(pdbid, m)
            t = G.tanimoto_bits(pb_, cbits)
            pt.append(t); pr_.append("" if not np.isnan(t) else "both_fingerprints_empty"); nb.append(len(pb_))
        except Exception as exc:
            pt.append(np.nan); pr_.append("pose_fingerprint_error:" + str(exc)[:60]); nb.append(-1)
    res = pd.concat([df.drop(columns=["path"]), bust.astype("boolean"), v], axis=1)
    res["rmsd"], res["rmsd_reason"], res["plif_tanimoto"], res["plif_reason"], res["n_bits_pose"] = rm, rr, pt, pr_, nb
    res["n_bits_crystal"] = -1 if cbits is None else len(cbits)
    res["plif_receptor_route"] = G.ROUTE.get(pdbid, "none")
    res["pb_timeout"] = [i in set(info["timeout_idx"]) for i in range(len(res))]
    tmp = Path(str(out) + ".tmp")
    res.to_parquet(tmp); tmp.rename(out)
    return pdbid, len(res), time.time() - t0, "done"


def main():
    pdbids = sorted(p.stem for p in RAW.glob("*.jsonl"))
    if LIMIT:
        pdbids = pdbids[:LIMIT]
    print("[%s] complexes with docking records: %d | nproc %d" % (time.strftime("%H:%M:%S"), len(pdbids), NPROC), flush=True)
    t0 = time.time(); n = 0
    with Pool(NPROC, maxtasksperchild=10) as pool:
        for pdbid, k, sec, st in pool.imap_unordered(work, pdbids):
            n += 1
            print("[%s] %d/%d %s %s poses %d (%.0f s)" % (time.strftime("%H:%M:%S"), n, len(pdbids), pdbid, st, k, sec), flush=True)
    print("[%s] finished in %.1f min" % (time.strftime("%H:%M:%S"), (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
