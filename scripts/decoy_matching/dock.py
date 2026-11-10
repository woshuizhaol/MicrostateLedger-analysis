import csv, fcntl, gzip, hashlib, json, os, shutil, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_pipeline as PL

ROOT = PL.ROOT
CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
D = CFG["docking"]
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
DOCK = OUT_ROOT / "raw_outputs/decoy_matching/dock"; DOCK.mkdir(parents=True, exist_ok=True)
SCR = OUT_ROOT / "raw_outputs/decoy_matching/dock_scratch"; SCR.mkdir(parents=True, exist_ok=True)
QUEUE = Path(os.environ.get("DECOY_QUEUE", str(ROOT / "raw_outputs/decoy_matching/queue/ligand_sets.csv")))
RECEPTORS = ROOT / "inputs/receptors"
NPROC = int(os.environ.get("DECOY_NPROC", str(D["nproc"])))
MAX_ROUND = int(os.environ.get("DECOY_MAX_ROUND", "3"))
LIMIT = int(os.environ.get("DECOY_LIMIT", "0"))
SEED_BASE = int(D["seed_base"])
STOP = DOCK / "STOP"
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as _P
assert _P.file_md5(PL.VINA) == CFG["pinned_md5"]["vina"], "Vina binary changed"
assert _P.file_md5(PL.CONF_SCRIPT) == CFG["pinned_md5"]["conformer_script"], "conformer script changed"


def uid_of(target, smiles):
    return hashlib.md5(("%s|%s" % (target, smiles)).encode()).hexdigest()[:12]


def receptor_for(t):
    d = RECEPTORS / (t if t.startswith("litpcba_") else "dude_" + t)
    return d / "receptor.pdbqt", json.loads((d / "box_from_crystal_ligand.json").read_text())


def top1_record(sdf_path, title):
    txt = Path(sdf_path).read_text()
    rec = txt.split("$$$$")[0]
    lines = rec.split("\n")
    lines[0] = title
    return "\n".join(lines).rstrip("\n") + "\n$$$$\n"


def work(task):
    target, smiles, uid, rnd = task
    if STOP.exists():
        return None, []
    wd = SCR / uid
    t0 = time.time()
    res = dict(uid=uid, target=target, smiles=smiles, round=rnd, status="error", error="", n_eligible=0, n_all=0, microstates=[], std_smiles="", host_pid=os.getpid())
    poses = []
    try:
        shutil.rmtree(wd, ignore_errors=True)
        e = PL.enumerate_ligand(smiles, out_dir=str(wd / "enum"))
        res.update(std_smiles=e["std_smiles"], n_all=len(e["all"]), n_selected=len(e["selected"]), n_eligible=len(e["eligible"]), dimorphite=len(e["dimorphite"]))
        if not e["eligible"]:
            res["status"] = "no_microstate"
        else:
            rec_path, box = receptor_for(target)
            for k, rec in enumerate(e["eligible"]):
                r = PL.dock_microstate(rec, rec_path, box, wd / "dock", SEED_BASE)
                item = dict(order=k, microstate_id=rec["microstate_id"], formal_charge=rec["formal_charge"], smiles=rec["stereo_tag"], tautomer_key=rec["tautomer_key"], status=r["status"], stage=r.get("stage", ""),
                            seed=r["seed"], n_attempts=r.get("n_attempts"), failure_by_stage=r.get("failure_by_stage", {}), n_conformers=r.get("n_conformers"),
                            cpu_conf=r.get("cpu_conf"), cpu_prep=r.get("cpu_prep"), cpu_vina=r.get("cpu_vina"), wall_total=r.get("wall_total"))
                if r["status"] == "ok":
                    item.update(score=r["score"], mode_scores=r["mode_scores"], conf_rank=r["conf_rank"], conf_energy=r["conf_energy"])
                    poses.append(top1_record(r["pose_sdf"], "%s|%s" % (uid, rec["microstate_id"])))
                else:
                    item["error"] = r.get("error", "")
                res["microstates"].append(item)
            n_ok = sum(1 for m in res["microstates"] if m["status"] == "ok")
            res["status"] = "ok" if n_ok == len(res["microstates"]) else ("partial" if n_ok else "all_failed")
    except Exception as exc:
        res["error"] = str(exc)[-200:]
    shutil.rmtree(wd, ignore_errors=True)
    res["wall_seconds"] = time.time() - t0
    res["cpu_seconds"] = sum((m.get("cpu_conf") or 0) + (m.get("cpu_prep") or 0) + (m.get("cpu_vina") or 0) for m in res["microstates"])
    return res, poses


def done_uids():
    s = set()
    for f in DOCK.glob("*.jsonl"):
        if f.name == "errors.jsonl":
            continue
        for ln in open(f, encoding="utf-8"):
            try:
                s.add(json.loads(ln)["uid"])
            except Exception:
                pass
    return s


def tasks():
    seen, out = {}, []
    for r in csv.DictReader(open(QUEUE, encoding="utf-8")):
        rnd = int(r["round"])
        if rnd > MAX_ROUND:
            continue
        key = (r["target"], r["smiles"])
        if key in seen:
            continue
        seen[key] = 1
        out.append((r["target"], r["smiles"], uid_of(*key), rnd))
    return out


def gen(todo):
    for t in todo:
        if STOP.exists():
            print("[%s] STOP file found -- no more tasks are submitted" % time.strftime("%H:%M:%S"), flush=True)
            return
        yield t


if __name__ == "__main__":
    lock = open(DOCK / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another dock process holds the lock")
    allt = tasks()
    done = done_uids()
    todo = [t for t in allt if t[2] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print("[%s] tasks %d, done %d, to do %d, nproc %d, max round %d" % (time.strftime("%H:%M:%S"), len(allt), len(done & {t[2] for t in allt}), len(todo), NPROC, MAX_ROUND), flush=True)
    t0 = time.time(); n = 0; cpu = 0.0; nfail = 0; nms = 0; nerr = 0
    handles = {}
    with Pool(NPROC, maxtasksperchild=40) as pool:
        for res, poses in pool.imap_unordered(work, gen(todo), chunksize=1):
            if res is None:
                continue
            if res["status"] == "error":
                with open(DOCK / "errors.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                nerr += 1
                continue
            t = res["target"]
            if poses:
                blob = gzip.compress("".join(poses).encode())
                with open(DOCK / (t + ".poses.sdf.gz"), "ab") as f:
                    f.write(blob)
            with open(DOCK / (t + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            n += 1; cpu += res["cpu_seconds"]; nms += len(res["microstates"]); nfail += res["status"] not in ("ok",)
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                print("[%s] %d/%d ligands (%d microstates) | not-ok %d | exceptions %d | %.1f h elapsed | CPU %.1f h | eta %.1f h" % (time.strftime("%H:%M:%S"), n, len(todo), nms, nfail, nerr, el / 3600, cpu / 3600,
                      (len(todo) - n) * el / max(n, 1) / 3600), flush=True)
    print("[%s] finished %d ligands in %.1f h" % (time.strftime("%H:%M:%S"), n, (time.time() - t0) / 3600), flush=True)
