import gzip, io, os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import posebusters_runner as PB
from rdkit import Chem

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
DECOY_DIR = Path(os.environ.get("DECOY_ROOT", str(ROOT)))
RAW = OUT / "raw_outputs/pose_validity"
NPROC = int(os.environ.get("POSE_NPROC", "4")); CHUNK = int(os.environ.get("POSE_CHUNK", "64")); LIMIT = int(os.environ.get("POSE_LIMIT", "0"))
CHUNK_CPU = int(os.environ.get("POSE_CHUNK_CPU", "2400")); POSE_CPU = int(os.environ.get("POSE_POSE_CPU", "600"))


def run_chunk(job):
    out_path, receptor, items = job
    mols = []
    for pid, src in items:
        try:
            if "\n" in src or src.endswith("$$$$"):
                m = next((x for x in Chem.ForwardSDMolSupplier(io.BytesIO(src.encode()), removeHs=False)), None)
            else:
                m = Chem.SDMolSupplier(src, removeHs=False)[0]
        except Exception:
            m = None
        mols.append(m)
    df, info = PB.bust_mols_guarded(mols, receptor, chunk_cpu=CHUNK_CPU, pose_cpu=POSE_CPU)
    if info["timeout_idx"]:
        with open(RAW / "bust" / "pb_timeouts.tsv", "a", encoding="utf-8") as f:
            for i in info["timeout_idx"]:
                f.write("%s\t%s\t%d\t%s\n" % (items[i][0], Path(out_path).parent.name + "/" + Path(out_path).name, POSE_CPU, time.strftime("%F %T")))
    v = PB.verdicts(df)
    res = pd.concat([pd.DataFrame({"pose_id": [p for p, _ in items]}), df.astype("boolean"), v], axis=1)
    tmp = Path(str(out_path) + ".tmp")
    res.to_parquet(tmp); tmp.rename(out_path)
    return str(out_path), len(items), int(v["full_pb"].sum())


def ten_target_jobs():
    man = pd.read_csv(RAW / "ten_target_pose_manifest.csv", usecols=["pose_id", "dataset", "seed", "target", "sdf_path", "receptor_pdb"])
    d = RAW / "bust/original"; d.mkdir(parents=True, exist_ok=True)
    jobs = []
    names = set()
    for (ds, sd, t, rec), g in man.groupby(["dataset", "seed", "target", "receptor_pdb"], sort=True):
        items = list(zip(g.pose_id, g.sdf_path))
        for k in range(0, len(items), CHUNK):
            nm = "%s__%s__%s__%05d.parquet" % (ds, sd, t, k // CHUNK)
            assert nm not in names, "output name collision: " + nm
            names.add(nm)
            p = d / nm
            if not p.exists():
                jobs.append((str(p), rec, items[k:k + CHUNK]))
    return jobs


def full_scale_jobs(source="full_scale"):
    d = RAW / ("bust/" + source); d.mkdir(parents=True, exist_ok=True)
    done = {}
    for p in d.glob("*.parquet"):
        t = p.name.split("__")[0]
        done.setdefault(t, set()).update(pd.read_parquet(p, columns=["pose_id"]).pose_id.tolist())
    jobs = []
    src_dir = (DECOY_DIR / "raw_outputs/decoy_matching/dock") if source == "full_scale" else (RAW / "noise")
    for gz in sorted(src_dir.glob("*.poses.sdf.gz")):
        t = gz.name.split(".")[0]
        rec_dir = ROOT / "inputs/receptors" / (t if t.startswith("litpcba_") else "dude_" + t)
        rec = rec_dir / "receptor_bust.pdb"
        txt = []
        try:
            with gzip.open(gz, "rt") as f:
                blob = f.read()
        except EOFError:
            blob = ""
        recs = [r + "$$$$\n" for r in blob.split("$$$$\n") if r.strip()]
        items = []
        for r in recs:
            title = r.split("\n", 1)[0].strip()
            if title not in done.get(t, set()):
                items.append((title, r))
        n_exist = len(list(d.glob(t + "__*.parquet")))
        stamp = int(time.time())
        for k in range(0, len(items), CHUNK):
            jobs.append((str(d / ("%s__%d_%05d.parquet" % (t, stamp, n_exist + k // CHUNK))), str(rec), items[k:k + CHUNK]))
    return jobs


def main():
    src = sys.argv[sys.argv.index("--source") + 1]
    jobs = ten_target_jobs() if src == "ten_target" else full_scale_jobs(src)
    if LIMIT:
        jobs = jobs[:LIMIT]
    print("[%s] source %s: %d chunks to do (%d poses), %d workers" % (time.strftime("%H:%M:%S"), src, len(jobs), sum(len(j[2]) for j in jobs), NPROC), flush=True)
    t0 = time.time(); n = 0; npass = 0
    with Pool(NPROC, maxtasksperchild=20) as pool:
        for i, (p, k, ok) in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            n += k; npass += ok
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print("[%s] %d/%d chunks | %d poses | full_pb pass %.1f%% | %.1f min | eta %.1f h" % (time.strftime("%H:%M:%S"), i, len(jobs), n, 100 * npass / max(n, 1), el / 60, (len(jobs) - i) * el / i / 3600), flush=True)
    print("[%s] done" % time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
