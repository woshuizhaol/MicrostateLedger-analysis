import csv, hashlib, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "decoy_matching"))
import decoy_pipeline as PL
import numpy as np
from rdkit import Chem

ROOT, VF = PL.ROOT, PL.VF
RM = Path(os.environ.get("BASE_ROOT", "."))
OUT = Path(os.environ.get("CRITERIA_OUT_ROOT", str(ROOT)))
RAW = OUT / "raw_outputs/pose_criteria/redock"; RAW.mkdir(parents=True, exist_ok=True)
SCR = OUT / "raw_outputs/pose_criteria/redock_scratch"; SCR.mkdir(parents=True, exist_ok=True)
CORE = VF / "data/processed/CASF-2016/CASF-2016/coreset"
RECDIR = RM / "results/benchmark/p3_casf_scoring_power/receptors"
SEEDS = [20261041, 20261042, 20261043]
BASE = 20261015
NPROC = int(os.environ.get("CRITERIA_NPROC", "4")); LIMIT = int(os.environ.get("CRITERIA_LIMIT", "0")); SUBSET = os.environ.get("CRITERIA_SUBSET", "affected")
STOP = RAW / "STOP"


def box_of(pdbid):
    m = Chem.MolFromMol2File(str(CORE / pdbid / ("%s_ligand.mol2" % pdbid)), removeHs=False, sanitize=True)
    mh = Chem.RemoveHs(m); c = mh.GetConformer()
    xyz = np.array([list(c.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())])
    cen = xyz.mean(0); ext = xyz.max(0) - xyz.min(0)
    return dict(center=dict(zip("xyz", map(float, cen))), size=dict(zip("xyz", [float(max(22.5, e + 10)) for e in ext])))


def work(task):
    pdbid, mid, smi = task
    if STOP.exists():
        return None
    wd = SCR / ("%s_%s" % (pdbid, mid)); shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
    res = dict(pdbid=pdbid, microstate_id=mid, smiles=smi, status="error", seeds={}, error="")
    t0 = time.time()
    try:
        m = Chem.MolFromSmiles(smi)
        w = Chem.SDWriter(str(wd / "ms.sdf")); w.write(m); w.close()
        box = box_of(pdbid); rec = RECDIR / ("%s.pdbqt" % pdbid)
        cseed = PL.microstate_seed(BASE, "%s|%s" % (pdbid, mid))
        confs, payload = PL.make_conformers(wd / "ms.sdf", wd / "conf", cseed)
        if not confs:
            res.update(status="failed", error="no_conformer"); raise StopIteration
        pdir = RAW / pdbid; pdir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for s in SEEDS:
            r = dict(status="failed", failure_by_stage={})
            fail = {}
            for rank, (csdf, en) in enumerate(confs):
                try:
                    lig = PL.prepare_ligand(csdf, wd / ("lig_%d.pdbqt" % rank))
                except Exception:
                    fail["dockprep_failed"] = fail.get("dockprep_failed", 0) + 1; continue
                try:
                    outp = wd / ("dock_%d_%d.pdbqt" % (rank, s))
                    sc = PL.vina_dock(rec, lig, box["center"], box["size"], s, outp)
                except Exception:
                    fail["vina_failed"] = fail.get("vina_failed", 0) + 1; continue
                try:
                    sdf = PL.export_pose_sdf(outp, wd / ("dock_%d_%d.sdf" % (rank, s)))
                except Exception:
                    fail["pose_ingest_failed"] = fail.get("pose_ingest_failed", 0) + 1; continue
                txt = Path(sdf).read_text().split("$$$$")[0].rstrip("\n") + "\n$$$$\n"
                (pdir / ("%s_%d.sdf" % (mid, s))).write_text(txt)
                r = dict(status="ok", score=sc[0], mode_scores=sc, conf_rank=rank, conf_energy=en, failure_by_stage=fail); ok += 1
                break
            else:
                r["failure_by_stage"] = fail
            res["seeds"][str(s)] = r
        res.update(status="ok" if ok == len(SEEDS) else ("partial" if ok else "all_failed"), n_conformers=len(confs), conformer_seed=cseed)
    except StopIteration:
        pass
    except Exception as exc:
        res["error"] = str(exc)[-160:]
    shutil.rmtree(wd, ignore_errors=True)
    res["wall_seconds"] = time.time() - t0
    return res


def main():
    ms = list(csv.DictReader(open(ROOT / "raw_outputs/casf_scoring/per_microstate_inplace_scores.csv", encoding="utf-8")))
    if SUBSET == "affected":
        ps = [r for r in csv.DictReader(open(ROOT / "raw_outputs/casf_scoring/predictor_states.csv", encoding="utf-8")) if r["predictor"] == "dimorphite_dom" and r["differs_from_single"] == "True"]
        keep = {r["pdbid"] for r in ps}
    elif SUBSET == "all":
        keep = {r["pdbid"] for r in ms}
    else:
        keep = set(SUBSET.split(","))
    done = set()
    for f in RAW.glob("*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln); done.add((r["pdbid"], r["microstate_id"]))
    tasks = [(r["pdbid"], r["microstate_id"], r["smiles"]) for r in ms if r["pdbid"] in keep and (r["pdbid"], r["microstate_id"]) not in done]
    if LIMIT:
        tasks = tasks[:LIMIT]
    print("[%s] complexes %d | microstates to do %d (done %d) | nproc %d" % (time.strftime("%H:%M:%S"), len(keep), len(tasks), len(done), NPROC), flush=True)
    t0 = time.time(); n = 0
    with Pool(NPROC, maxtasksperchild=25) as pool:
        for res in pool.imap_unordered(work, tasks, chunksize=1):
            if res is None:
                continue
            with open(RAW / ("%s.jsonl" % res["pdbid"]), "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            n += 1
            if n % 50 == 0 or n == len(tasks):
                print("[%s] %d/%d microstates | %.1f h | eta %.1f h" % (time.strftime("%H:%M:%S"), n, len(tasks), (time.time() - t0) / 3600, (len(tasks) - n) * (time.time() - t0) / n / 3600), flush=True)
    print("[%s] finished %d" % (time.strftime("%H:%M:%S"), n), flush=True)


if __name__ == "__main__":
    main()
