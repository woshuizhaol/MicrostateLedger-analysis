import io, os, pickle, resource, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from posebusters import PoseBusters

META = ("file", "molecule", "position")
GEOM = ["mol_pred_loaded", "sanitization", "inchi_convertible", "all_atoms_connected", "no_radicals", "bond_lengths", "bond_angles", "internal_steric_clash",
        "aromatic_ring_flatness", "non-aromatic_ring_non-flatness", "double_bond_flatness", "internal_energy"]
INTER = ["mol_cond_loaded", "protein-ligand_maximum_distance", "minimum_distance_to_protein", "minimum_distance_to_organic_cofactors", "minimum_distance_to_inorganic_cofactors",
         "minimum_distance_to_waters", "volume_overlap_with_protein", "volume_overlap_with_organic_cofactors", "volume_overlap_with_inorganic_cofactors", "volume_overlap_with_waters"]
ALL = GEOM + INTER
_PB = {}


def pb(skip_energy=False):
    key = bool(skip_energy)
    if key not in _PB:
        import copy, yaml
        import posebusters
        cfg = copy.deepcopy(yaml.safe_load(open(Path(posebusters.__file__).parent / "config" / "dock.yml")))
        mods = []
        for m in cfg["modules"]:
            if m["function"] == "energy_ratio":
                if skip_energy:
                    continue
                m["parameters"]["num_threads"] = 1
            mods.append(m)
        cfg["modules"] = mods
        _PB[key] = PoseBusters(config=cfg, top_n=1)
    return _PB[key]


def bust_mols(mols, receptor_pdb, skip_energy=False):
    ok = [i for i, m in enumerate(mols) if m is not None]
    rows = {}
    if ok:
        res = pb(skip_energy).bust([mols[i] for i in ok], None, str(receptor_pdb))
        res = res.reset_index(drop=True)
        if skip_energy and "internal_energy" not in res.columns:
            res["internal_energy"] = np.nan
        missing = [c for c in ALL if c not in res.columns]
        if missing:
            raise RuntimeError("PoseBusters columns missing: %s" % missing)
        for k, i in enumerate(ok):
            rows[i] = res.loc[k, ALL]
    out = []
    for i in range(len(mols)):
        out.append(rows[i] if i in rows else pd.Series({c: (False if c == "mol_pred_loaded" else np.nan) for c in ALL}))
    return pd.DataFrame(out).reset_index(drop=True)


def verdicts(df):
    t = df.map(lambda v: v is True or (isinstance(v, (bool, np.bool_)) and bool(v)))
    nan = df.isna()
    full = t[ALL].all(axis=1)
    geom = t[GEOM].all(axis=1)
    inter = t[INTER].all(axis=1)
    failed = t[ALL].apply(lambda r: ";".join(c for c in ALL if not r[c]), axis=1)
    nan_cols = nan.apply(lambda r: ";".join(c for c in ALL if r[c]), axis=1)
    return pd.DataFrame(dict(full_pb=full, geom=geom, inter=inter, failed_checks=failed, pb_nan_columns=nan_cols))


def _run_forked(fn, cpu_limit):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 3
        try:
            os.close(r)
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_limit), int(cpu_limit) + 5))
            data = pickle.dumps(fn(), protocol=4)
            with os.fdopen(w, "wb") as f:
                f.write(data)
            code = 0
        finally:
            os._exit(code)
    os.close(w)
    parts = []
    with os.fdopen(r, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            parts.append(b)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and parts:
        return True, pickle.loads(b"".join(parts))
    return False, None


def bust_mols_guarded(mols, receptor_pdb, chunk_cpu=2400, pose_cpu=600):
    ok, df = _run_forked(lambda: bust_mols(mols, receptor_pdb), chunk_cpu)
    if ok:
        return df, dict(fallback=False, timeout_idx=[])
    rows, tout = [], []
    for i, m in enumerate(mols):
        ok1, d1 = _run_forked(lambda m=m: bust_mols([m], receptor_pdb), pose_cpu)
        if ok1:
            rows.append(d1.iloc[0])
            continue
        ok2, d2 = _run_forked(lambda m=m: bust_mols([m], receptor_pdb, skip_energy=True), 300)
        row = d2.iloc[0].copy() if ok2 else pd.Series({c: (False if c == "mol_pred_loaded" else np.nan) for c in ALL})
        row["internal_energy"] = np.nan
        rows.append(row)
        tout.append(i)
    return pd.DataFrame(rows).reset_index(drop=True), dict(fallback=True, timeout_idx=tout)


def read_sdf_text(text):
    sup = Chem.ForwardSDMolSupplier(io.BytesIO(text.encode()), removeHs=False)
    mols, titles = [], []
    for m in sup:
        mols.append(m)
        titles.append(m.GetProp("_Name") if m is not None and m.HasProp("_Name") else "")
    return mols, titles
