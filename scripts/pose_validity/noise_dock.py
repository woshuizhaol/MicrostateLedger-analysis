import gzip, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "decoy_matching"))
import decoy_pipeline as PL
import pandas as pd

ROOT = PL.ROOT
CFG = yaml.safe_load(open(ROOT / "configs/pose_validity.yaml", encoding="utf-8"))["noise_floor"]
OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RAW = OUT / "raw_outputs/pose_validity/noise"; RAW.mkdir(parents=True, exist_ok=True)
SCR = OUT / "raw_outputs/pose_validity/noise_scratch"; SCR.mkdir(parents=True, exist_ok=True)
RECEPTORS = ROOT / "inputs/receptors"
NPROC = int(os.environ.get("POSE_NPROC", "3")); LIMIT = int(os.environ.get("POSE_LIMIT", "0"))
SEEDS = [int(CFG["seed_base"]) + k for k in range(1, int(CFG["n_extra_seeds"]) + 1)]
STOP = RAW / "STOP"


def receptor_for(t):
    d = RECEPTORS / (t if t.startswith("litpcba_") else "dude_" + t)
    return d / "receptor.pdbqt", json.loads((d / "box_from_crystal_ligand.json").read_text())


def work(row):
    if STOP.exists():
        return None, []
    uid, target, smiles = row["uid"], row["target"], row["smiles"]
    wd = SCR / uid; shutil.rmtree(wd, ignore_errors=True); wd.mkdir(parents=True)
    res = dict(uid=uid, target=target, set=row["set"], status="error", error="", units=[])
    poses = []
    t0 = time.time()
    try:
        e = PL.enumerate_ligand(smiles, out_dir=str(wd / "enum"))
        byid = {m["microstate_id"]: m for m in e["eligible"]}
        rec_path, box = receptor_for(target)
        for label in ("default", "best"):
            mid, rank = row[label + "_microstate"], int(row[label + "_conf_rank"])
            if mid not in byid:
                res["units"].append(dict(label=label, microstate_id=mid, status="microstate_not_found")); continue
            seed0 = PL.microstate_seed(20261015, mid)
            confs, _ = PL.make_conformers(byid[mid]["sdf_path"], wd / ("conf_" + label), seed0)
            if rank >= len(confs):
                res["units"].append(dict(label=label, microstate_id=mid, status="rank_out_of_range", n_conformers=len(confs))); continue
            lig = PL.prepare_ligand(confs[rank][0], wd / (label + ".pdbqt"))
            e0 = row.get(label + "_conf_energy")
            u = dict(label=label, microstate_id=mid, conf_rank=rank, conf_energy=confs[rank][1], conf_energy_stage5=e0, conformer_reproduced=bool(e0 is not None and abs(confs[rank][1] - float(e0)) < 1e-6), seeds={})
            for s in SEEDS:
                try:
                    outp = wd / ("%s_%d.pdbqt" % (label, s))
                    sc = PL.vina_dock(rec_path, lig, box["center"], box["size"], s, outp)
                    sdf = PL.export_pose_sdf(outp, wd / ("%s_%d.sdf" % (label, s)))
                    txt = Path(sdf).read_text().split("$$$$")[0].split("\n", 1)[1]
                    poses.append("%s|%s|%d\n%s\n$$$$\n" % (uid, mid, s, txt.rstrip("\n")))
                    u["seeds"][str(s)] = dict(status="ok", score=sc[0])
                except Exception as exc:
                    u["seeds"][str(s)] = dict(status="failed", error=str(exc)[-80:])
            u["status"] = "ok" if all(v["status"] == "ok" for v in u["seeds"].values()) else "partial"
            res["units"].append(u)
        res["status"] = "ok"
    except Exception as exc:
        res["error"] = str(exc)[-160:]
    shutil.rmtree(wd, ignore_errors=True)
    res["wall_seconds"] = time.time() - t0
    return res, poses


def main():
    smp = pd.read_csv(OUT / "results/pose_validity/noise_sample.csv", dtype={"ligand_id": str})
    smp = smp.drop_duplicates(["uid"])
    done = set()
    for f in RAW.glob("*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            done.add(json.loads(ln)["uid"])
    todo = [r for r in smp.to_dict("records") if r["uid"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print("[%s] ligands to do %d (done %d) | seeds %s | nproc %d" % (time.strftime("%H:%M:%S"), len(todo), len(done), SEEDS, NPROC), flush=True)
    t0 = time.time(); n = 0
    with Pool(NPROC, maxtasksperchild=20) as pool:
        for res, poses in pool.imap_unordered(work, todo, chunksize=1):
            if res is None:
                continue
            if poses:
                with open(RAW / (res["target"] + ".poses.sdf.gz"), "ab") as f:
                    f.write(gzip.compress("".join(poses).encode()))
            with open(RAW / (res["target"] + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            n += 1
            if n % 10 == 0 or n == len(todo):
                print("[%s] %d/%d ligands | %.1f h | eta %.1f h" % (time.strftime("%H:%M:%S"), n, len(todo), (time.time() - t0) / 3600, (len(todo) - n) * (time.time() - t0) / n / 3600), flush=True)
    print("[%s] finished %d" % (time.strftime("%H:%M:%S"), n), flush=True)


if __name__ == "__main__":
    main()
