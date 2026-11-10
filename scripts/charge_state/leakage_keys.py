import csv, gzip, json, sys, time
import multiprocessing as mp
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charge_common as C
from rdkit import Chem

C.check_pins()
RAW = C.RAW; RAW.mkdir(parents=True, exist_ok=True)
NP = C.CFG["stats"]["threads"]


def key_smiles(args):
    smi, ik_given = args
    k = C.skeleton_key(Chem.MolFromSmiles(smi)) if smi else None
    return k, (ik_given[:14] if ik_given else None)


def write(name, keys, extra):
    keys = sorted(k for k in keys if k)
    with gzip.open(RAW / ("leak_keys_%s.txt.gz" % name), "wt") as f:
        f.write("\n".join(keys) + "\n")
    return dict(n_keys=len(keys), **extra)


def sdf_keys(path):
    out, n, bad = set(), 0, 0
    for m in Chem.SDMolSupplier(str(path)):
        n += 1
        if m is None:
            bad += 1; continue
        k = C.skeleton_key(m)
        if k: out.add(k)
        else: bad += 1
    return out, n, bad


t0 = time.time(); summ = {}
for name, key in (("molgpka_example", "molgpka_example_sdf"), ("pkasolver_train", "pkasolver_train_sdf")):
    ks, n, bad = sdf_keys(C.PATHS[key])
    summ[name] = write(name, ks, dict(source=str(C.PATHS[key]), n_records=n, n_unusable=bad))
    C.log(name, summ[name])
recs = []
with gzip.open(C.PATHS["chembl_chemreps"], "rt", encoding="utf-8") as f:
    rd = csv.DictReader(f, delimiter="\t")
    for r in rd:
        recs.append((r["canonical_smiles"], r["standard_inchi_key"]))
C.log("chembl rows read", len(recs))
keys, bad = set(), 0
with mp.get_context("fork").Pool(NP) as pool:
    for k, kg in pool.imap(key_smiles, recs, chunksize=2000):
        if k is None: bad += 1
        keys.add(k); keys.add(kg)
summ["chembl37"] = write("chembl37", keys, dict(source=str(C.PATHS["chembl_chemreps"]), n_records=len(recs), n_unusable=bad,
                                                note="union of the key of the cleaned structure and the first block of the registered standard InChIKey"))
C.log("chembl37", summ["chembl37"])
json.dump(dict(provenance=C.P.provenance_block(Path(__file__), C.ROOT / "configs/charge_state.yaml", [C.PATHS["molgpka_example_sdf"], C.PATHS["pkasolver_train_sdf"], C.PATHS["chembl_chemreps"]]),
               resources=summ, wall_seconds=round(time.time() - t0, 1)), open(RAW / "leak_keys_summary.json", "w"), indent=2)
