import csv, json, os, re, sqlite3, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
VF = Path(os.environ.get("DOCKING_ROOT", "."))
RM = Path(os.environ.get("BASE_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

OUT = Path(os.environ.get("POSE_OUT_ROOT", str(ROOT)))
RAW, RES = OUT / "raw_outputs/pose_validity", OUT / "results/pose_validity"; RAW.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
TARGETS = ["aa2ar", "adrb2", "bace1", "cdk2", "egfr", "fa10", "hivpr", "jak2", "pde5a", "src"]
RUN018 = VF / "verification/C2_dude_vs/outputs/ten_target_docking"
BUDGET6_DIR = RM / "results/benchmark/budget6_docking"
NAME_RE = re.compile(r"^(\d+)_(M-[0-9a-fA-F]+)_(\d+)\.sdf$")


def pick_receptor(tdir):
    for c in ("receptor_assets/receptor_filtered.pdb", "receptor_assets/receptor_for_meeko_primary_sanitize.pdb", "receptor_assets/receptor.pdbqt"):
        if (tdir / c).exists():
            return tdir / c
    raise FileNotFoundError(tdir)


def mode1(pdbqt):
    try:
        for ln in open(pdbqt):
            if ln.startswith("REMARK VINA RESULT:"):
                return float(ln.split()[3])
    except OSError:
        return None
    return None


def one(job):
    tag, seed, t, tdir = job
    tdir = Path(tdir)
    rows = []
    rec = pick_receptor(tdir)
    lig = {}
    for r in csv.DictReader(open(tdir / "rows.csv", encoding="utf-8")):
        lig[int(r["idx"])] = (r.get("cid", ""), r.get("label", ""), r.get("source", ""))
    ms = {}
    if (tdir / "ledger.sqlite").exists():
        con = sqlite3.connect("file:%s?mode=ro" % (tdir / "ledger.sqlite"), uri=True)
        for mid, cid, q, fx, st in con.execute("select microstate_id, compound_id, formal_charge, fixedh_inchi, stereo_tag from microstates"):
            ms[mid] = (cid, q, fx, st)
        con.close()
    n_noscore = 0
    for f in sorted((tdir / "tmp_vina_budget").glob("*.sdf")):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        idx, mid, rank = int(m.group(1)), m.group(2), int(m.group(3))
        sc = mode1(f.with_suffix(".pdbqt"))
        if sc is None:
            n_noscore += 1
        c = lig.get(idx, ("", "", ""))
        mm = ms.get(mid, ("", None, "", ""))
        rows.append(dict(pose_id="%s|%s|%d|%s|%d" % (tag, t, idx, mid, rank), dataset=tag, seed=seed, target=t, ligand_idx=idx, compound_id=c[0] or mm[0], label=c[1], source=c[2], microstate_id=mid, conf_rank=rank,
                         vina_score=sc, formal_charge=mm[1], fixedh_inchi=mm[2], stereo_tag=mm[3], sdf_path=str(f), receptor_pdb=str(rec)))
    return rows, n_noscore


def main():
    jobs = []
    for s in sorted(RUN018.glob("seed_*")):
        for t in TARGETS:
            jobs.append(("ten_target_main", s.name.split("_")[1], t, str(s / "docking_run" / t)))
    for s in sorted(BUDGET6_DIR.glob("seed_*")):
        for t in TARGETS:
            if (s / t / "tmp_vina_budget").is_dir():
                jobs.append(("ten_target_budget6", s.name.split("_")[1], t, str(s / t)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("POSE_NPROC", "4"))) as ex:
        res = list(ex.map(one, jobs))
    rows = [r for rr, _ in res for r in rr]
    df = pd.DataFrame(rows)
    out = RAW / "ten_target_pose_manifest.csv"
    cols = [("pose_id", "", "dataset|target|ligand idx|microstate|conformer rank"), ("dataset", "", "ten_target_main (ten-target docking) / ten_target_budget6 (budget-6 docking)"), ("seed", "", "run seed"), ("target", "", "DUD-E target"),
            ("ligand_idx", "", "ligand index in the run's rows.csv"), ("compound_id", "", "original ledger compound id"), ("label", "", "1 active, 0 decoy"), ("source", "", "active / decoy"), ("microstate_id", "", "original microstate id"),
            ("conf_rank", "", "energy rank of the conformer that was docked"), ("vina_score", "kcal/mol", "mode-1 Vina score (pdbqt REMARK VINA RESULT)"), ("formal_charge", "e", "formal charge of the microstate (original ledger)"),
            ("fixedh_inchi", "", "fixed-H InChI of the microstate (original ledger)"), ("stereo_tag", "", "isomeric SMILES of the microstate (original ledger)"), ("sdf_path", "", "pose file (mk_export of the Vina output; mode 1 is the first record)"),
            ("receptor_pdb", "", "receptor used by PoseBusters (as in the original analysis)")]
    df[[c[0] for c in cols]].to_csv(out, index=False)
    block = P.provenance_block(Path(__file__), None, [], seed=None, extra=dict(runs=[j[0] + ":" + j[1] + ":" + j[2] for j in jobs][:3], n_jobs=len(jobs)))
    P.write_schema(out, cols, "one row per original docked pose file", "Manifest of the original poses (ten-target docking and budget6_docking) re-checked in pose-validity", extra=dict(provenance=block))
    summ = dict(n_poses=len(df), by_dataset=df.groupby("dataset").size().to_dict(), by_dataset_seed=df.groupby(["dataset", "seed"]).size().astype(int).rename(lambda x: str(x)).to_dict() if False else {"%s/%s" % k: int(v) for k, v in df.groupby(["dataset", "seed"]).size().items()},
                n_without_score=int(sum(n for _, n in res)), n_microstate_units=int(df.drop_duplicates(["dataset", "seed", "target", "ligand_idx", "microstate_id"]).shape[0]), seconds=round(time.time() - t0, 1))
    json.dump(summ, open(RES / "ten_target_pose_manifest_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
