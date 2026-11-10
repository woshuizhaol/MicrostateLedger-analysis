import os
import hashlib, importlib.util, json, os, re, resource, shutil, subprocess, sys, tempfile, time
from pathlib import Path

sys.dont_write_bytecode = True
VF = Path(os.environ.get("DOCKING_ROOT", "."))
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
ENV = VF / "envs/all"
PY = str(ENV / "bin/python")
VINA = str(ENV / "bin/vina")
MK_PREP_LIG = str(ENV / "bin/mk_prepare_ligand.py")
MK_EXPORT = str(ENV / "bin/mk_export.py")
SCRIPTS = VF / "MicrostateLedger/msl/resources/scripts"
VINA_MD5 = "f0fc6b2586d2bd4416030eb56b22e946"

from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, QED, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


E = _load("msl_rdkit_enum_microstates", SCRIPTS / "rdkit_enum_microstates.py")

PH_MIN, PH_MAX, MAX_VARIANTS = 6.4, 8.4, 32
MAX_TAUTOMERS, MAX_ISOMERS = 32, 16
MAX_MICROSTATES, KEEP_ALL_IF_TOTAL, TOPK = 64, 32, 16
MAX_ABS_CHARGE = 2
N_CONF, PRUNE_RMSD = 4, 0.5


def ingest(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unparsable")
    mol = E._clear_atom_map_numbers(mol)
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.ChargeParent(mol)
    te = rdMolStandardize.TautomerEnumerator()
    mol = te.Canonicalize(mol)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "ingest.sdf")
        w = Chem.SDWriter(f); w.write(mol); w.close()
        m2 = next((m for m in Chem.SDMolSupplier(f, removeHs=False) if m is not None), None)
    if m2 is None:
        raise ValueError("sdf roundtrip failed")
    return Chem.MolToSmiles(E._clear_atom_map_numbers(m2), isomericSmiles=True)


def dimorphite_smiles(smiles):
    from dimorphite_dl import protonate_smiles
    out = protonate_smiles(smiles, ph_min=PH_MIN, ph_max=PH_MAX, max_variants=MAX_VARIANTS)
    return sorted(set(s.split()[0] for s in out if s and s.strip()))


def enumerate_records(smiles_list, out_dir=None):
    te = rdMolStandardize.TautomerEnumerator(); te.SetMaxTautomers(MAX_TAUTOMERS)
    uncharger = rdMolStandardize.Uncharger()
    opts = StereoEnumerationOptions(tryEmbedding=False, unique=True, onlyUnassigned=True)
    seen, out = set(), []
    stats = dict(n_input_smiles=len(smiles_list), tautomer_truncated=0, stereo_truncated=0, sanitize_failed=0, invalid_smiles=0)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_smiles"] += 1
            continue
        taut_by_key = {}
        for taut in list(te.Enumerate(mol)):
            ct = te.Canonicalize(taut)
            taut_by_key.setdefault(Chem.MolToSmiles(ct, isomericSmiles=True), Chem.Mol(taut))
        titems = sorted(taut_by_key.items(), key=lambda it: it[0])
        if len(titems) > MAX_TAUTOMERS:
            stats["tautomer_truncated"] += len(titems) - MAX_TAUTOMERS
            titems = titems[:MAX_TAUTOMERS]
        for tkey, taut in titems:
            stereo_raw = list(EnumerateStereoisomers(taut, options=opts)) or [Chem.Mol(taut)]
            sb = {}
            for iso in stereo_raw:
                try:
                    sb.setdefault(Chem.MolToSmiles(iso, isomericSmiles=True), Chem.Mol(iso))
                except Exception:
                    continue
            sitems = sorted(sb.items(), key=lambda it: it[0])
            if len(sitems) > MAX_ISOMERS:
                stats["stereo_truncated"] += len(sitems) - MAX_ISOMERS
                sitems = sitems[:MAX_ISOMERS]
            for _k, iso in sitems:
                iso = E._clear_atom_map_numbers(iso)
                iso = E._standardize_candidate(iso)
                try:
                    Chem.SanitizeMol(iso)
                except Exception:
                    stats["sanitize_failed"] += 1
                    continue
                try:
                    fixedh = Chem.MolToInchi(Chem.AddHs(iso, addCoords=False), options="/FixedH")
                    stereo_tag = Chem.MolToSmiles(iso, isomericSmiles=True)
                except Exception:
                    continue
                charge = Chem.GetFormalCharge(iso)
                try:
                    parent = uncharger.uncharge(rdMolStandardize.ChargeParent(iso))
                    pkey = Chem.MolToSmiles(parent, isomericSmiles=True)
                except Exception:
                    pkey = stereo_tag
                try:
                    hbd, hba, qed = int(Lipinski.NumHDonors(iso)), int(Lipinski.NumHAcceptors(iso)), float(QED.qed(iso))
                except Exception:
                    hbd, hba, qed = 0, 0, 0.0
                sig = E.coordination_signature(iso)
                has_metal = E._has_metal_atoms(iso)
                q_abs = abs(int(charge))
                dock_eligible = False if has_metal else bool(q_abs <= 2)
                mid = E.microstate_id(fixedh, charge, stereo_tag, sig)
                if mid in seen:
                    continue
                seen.add(mid)
                mol_out, mapping = E.canonicalize_atom_mapping(Chem.Mol(iso))
                rec = dict(microstate_id=mid, fixedh_inchi=fixedh, formal_charge=int(charge), stereo_tag=stereo_tag, coordination_sig=sig, protonation_key=pkey, tautomer_key=tkey,
                           hbd=hbd, hba=hba, rdkit_score=qed, abs_formal_charge=q_abs, has_metal=bool(has_metal), dock_eligible=dock_eligible,
                           dock_priority_score=float(E._dock_priority_score(int(charge), qed, bool(has_metal))))
                if out_dir:
                    os.makedirs(os.path.join(out_dir, mid), exist_ok=True)
                    rec_path = os.path.join(out_dir, mid, "microstate.sdf")
                    w = Chem.SDWriter(rec_path); w.write(mol_out); w.close()
                    rec["sdf_path"] = rec_path
                    with open(os.path.join(out_dir, mid, "atom_map.json"), "w", encoding="utf-8") as fh:
                        fh.write(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True))
                else:
                    rec["sdf_path"] = ""
                out.append(rec)
    out = sorted(out, key=lambda m: (m["microstate_id"], m["stereo_tag"]))
    return out, stats


def select_microstates(recs):
    def rank_key(m):
        return (0.0 if m.get("dock_eligible", True) else 100.0, 1.0 if m.get("has_metal", False) else 0.0, abs(int(m["formal_charge"])), abs(int(m["hbd"]) - int(m["hba"])), -float(m["rdkit_score"]))
    if len(recs) > KEEP_ALL_IF_TOTAL:
        return sorted(recs, key=rank_key)[:min(MAX_MICROSTATES, TOPK)]
    return list(recs)


def docking_order(selected):
    ordered = sorted(selected, key=lambda m: (abs(int(m["formal_charge"])), m["microstate_id"]))
    elig = [m for m in ordered if abs(int(m["formal_charge"])) <= MAX_ABS_CHARGE]
    return ordered, (elig or list(ordered))


def enumerate_ligand(smiles, out_dir=None, std=None):
    std = std if std is not None else ingest(smiles)
    dl = dimorphite_smiles(std)
    recs, stats = enumerate_records(dl, out_dir=out_dir)
    sel = select_microstates(recs)
    ordered, elig = docking_order(sel)
    return dict(std_smiles=std, dimorphite=dl, all=recs, selected=ordered, eligible=elig, stats=stats)


def run(cmd, timeout=600, cwd=None, env=None):
    cp = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, cwd=cwd, env=env)
    if cp.returncode != 0:
        raise RuntimeError("cmd failed: %s\nSTDOUT:%s\nSTDERR:%s" % (" ".join(cmd[:4]), cp.stdout[-400:], cp.stderr[-400:]))
    return (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")


CONF_SCRIPT = ROOT / "scripts/decoy_matching/microstate_modules/rdkit_conformers.py"
CONF_SCRIPT_MD5 = "ba263ba11b7e676b8e808bc424f96fd1"


def make_conformers(sdf_path, out_dir, seed):
    cmd = [PY, "-B", str(CONF_SCRIPT), "--input", str(sdf_path), "--out-dir", str(out_dir), "--n", str(N_CONF), "--prune_rmsd", str(PRUNE_RMSD), "--select-k", "10", "--cluster-threshold", str(PRUNE_RMSD), "--seed", str(seed)]
    out = run(cmd, timeout=1200)
    i = out.find("{")
    payload, _ = json.JSONDecoder().raw_decode(out[i:])
    recs = payload.get("conformers", [])
    return sorted([(r["sdf_path"], float(r["energy"])) for r in recs], key=lambda t: (t[1], t[0])), payload


def prepare_ligand(conf_sdf, out_pdbqt):
    run([PY, "-B", MK_PREP_LIG, "-i", str(conf_sdf), "-o", str(out_pdbqt)], timeout=600)
    return out_pdbqt


def vina_dock(receptor_pdbqt, ligand_pdbqt, center, size, seed, out_pdbqt, cpu=1):
    cmd = [VINA, "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt), "--center_x", "%.6f" % center["x"], "--center_y", "%.6f" % center["y"], "--center_z", "%.6f" % center["z"],
           "--size_x", "%.6f" % size["x"], "--size_y", "%.6f" % size["y"], "--size_z", "%.6f" % size["z"], "--seed", str(seed), "--exhaustiveness", "8", "--cpu", str(cpu), "--out", str(out_pdbqt)]
    txt = run(cmd, timeout=3600)
    scores = []
    for line in txt.splitlines():
        t = line.strip().split()
        if len(t) >= 2 and t[0].isdigit():
            try:
                scores.append(float(t[1]))
            except ValueError:
                pass
    if not scores:
        raise RuntimeError("cannot parse vina output")
    return scores


def export_pose_sdf(vina_pdbqt, out_sdf):
    run([PY, "-B", MK_EXPORT, "-s", str(out_sdf), str(vina_pdbqt)], timeout=300)
    return out_sdf


def microstate_seed(base_seed, mid):
    return int(base_seed) + int(hashlib.sha256(mid.encode()).hexdigest()[:8], 16) % 100000


def _ru():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def dock_microstate(rec, receptor, box, work, base_seed):
    mid = rec["microstate_id"]
    seed = microstate_seed(base_seed, mid)
    wd = Path(work) / mid; wd.mkdir(parents=True, exist_ok=True)
    fail = {}
    tm = dict(cpu_conf=0.0, cpu_prep=0.0, cpu_vina=0.0, wall_total=0.0)
    t_start = time.time()
    c0 = _ru()
    try:
        confs, payload = make_conformers(rec["sdf_path"], wd / "conf", seed)
    except Exception as exc:
        tm["cpu_conf"] = _ru() - c0; tm["wall_total"] = time.time() - t_start
        return dict(status="failed", stage="conformer_failed", microstate_id=mid, seed=seed, n_attempts=0, failure_by_stage={"conformer_failed": 1}, error=str(exc)[-200:], **tm)
    tm["cpu_conf"] = _ru() - c0
    if not confs:
        tm["wall_total"] = time.time() - t_start
        return dict(status="failed", stage="conformer_failed", microstate_id=mid, seed=seed, n_attempts=0, failure_by_stage={"conformer_failed": 1}, error="no_conformer", **tm)
    attempts = 0
    for rank, (csdf, energy) in enumerate(confs):
        attempts += 1
        c1 = _ru()
        try:
            lig = prepare_ligand(csdf, wd / ("lig_%d.pdbqt" % rank))
        except Exception as exc:
            tm["cpu_prep"] += _ru() - c1
            fail["dockprep_failed"] = fail.get("dockprep_failed", 0) + 1
            continue
        tm["cpu_prep"] += _ru() - c1
        c2 = _ru()
        try:
            outp = wd / ("dock_%d.pdbqt" % rank)
            scores = vina_dock(receptor, lig, box["center"], box["size"], seed, outp)
        except Exception as exc:
            tm["cpu_vina"] += _ru() - c2
            fail["vina_failed"] = fail.get("vina_failed", 0) + 1
            continue
        tm["cpu_vina"] += _ru() - c2
        try:
            sdf = export_pose_sdf(outp, wd / ("dock_%d.sdf" % rank))
        except Exception as exc:
            fail["pose_ingest_failed"] = fail.get("pose_ingest_failed", 0) + 1
            continue
        tm["wall_total"] = time.time() - t_start
        return dict(status="ok", microstate_id=mid, seed=seed, score=scores[0], mode_scores=scores, conf_rank=rank, conf_energy=energy, n_conformers=len(confs), n_attempts=attempts,
                    failure_by_stage=fail, pose_sdf=str(sdf), docked_pdbqt=str(outp), **tm)
    tm["wall_total"] = time.time() - t_start
    return dict(status="failed", stage=max(fail, key=fail.get) if fail else "exhausted", microstate_id=mid, seed=seed, n_attempts=attempts, failure_by_stage=fail, n_conformers=len(confs), **tm)
