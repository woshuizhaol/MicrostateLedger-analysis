import csv, json, os, random, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoy_pipeline as PL
import decoy_props as PR
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT, VF = PL.ROOT, PL.VF
RM = Path(os.environ.get("BASE_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

CFG = yaml.safe_load(open(ROOT / "configs/decoy_matching.yaml", encoding="utf-8"))
OUT_ROOT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
QDIR = OUT_ROOT / "raw_outputs/decoy_matching/queue"; QDIR.mkdir(parents=True, exist_ok=True)
RES = OUT_ROOT / "results/decoy_matching"; RES.mkdir(parents=True, exist_ok=True)
DUDE = VF / "verification/C2_dude_vs/inputs/dude_all_targets"
AVE = RM / "data/raw/lit_pcba/AVE_unbiased"
AVE_DIR = {"aldh1": "ALDH1", "esr1_ant": "ESR1_ant", "fen1": "FEN1", "gba": "GBA", "kat2a": "KAT2A", "mapk1": "MAPK1", "mtorc1": "MTORC1", "pkm2": "PKM2", "tp53": "TP53", "vdr": "VDR"}
DT, LT = CFG["targets"]["dude"], CFG["targets"]["litpcba"]
LS = CFG["ligand_sets"]
NEG = 3
CLASSES = ["<=-1", "0", "+1", ">=+2"]
NLABEL = int(os.environ.get("DECOY_NLABEL", "70000"))
NPROC = int(os.environ.get("DECOY_NPROC", "24"))
assert P.file_md5(ROOT / "inputs/external/zinc20/zinc_pool.tsv") == CFG["pinned_md5"]["zinc_pool"]
LOG = dict(sets={}, notes=[])


def read_smi(p):
    return [(ln.split()[1], ln.split()[0]) for ln in open(p, encoding="utf-8") if len(ln.split()) >= 2]


def klass(q):
    return None if q is None else ("<=-1" if q <= -1 else ("0" if q == 0 else ("+1" if q == 1 else ">=+2")))


def split_rounds(neg, n_act):
    return [(neg[(k - 1) * n_act: k * n_act], k) for k in (1, 2, 3)]


rows = []


def add(set_name, target, role, rnd, lid, smi, info=""):
    rows.append(dict(set=set_name, target=target, role=role, round=rnd, ligand_id=lid, smiles=smi, info=info))


dude_act = {}
for i, t in enumerate(DT):
    act = read_smi(DUDE / t / "actives_final.ism"); dec = read_smi(DUDE / t / "decoys_final.ism")
    dude_act[t] = act
    rng = random.Random(LS["dude_full"]["seed"] + i)
    pick = rng.sample(dec, NEG * len(act))
    for lid, smi in act:
        add("dude_full", t, "active", 0, lid, smi)
    for neg, k in split_rounds(pick, len(act)):
        for lid, smi in neg:
            add("dude_full", t, "negative", k, lid, smi, "decoy")
    LOG["sets"]["dude_full/" + t] = dict(n_active=len(act), n_negative=len(pick), n_decoys_available=len(dec))
for i, t in enumerate(LT):
    act = sorted({x for k in ("T", "V") for x in read_smi(AVE / AVE_DIR[t] / ("active_%s.smi" % k))})
    ina = sorted({x for k in ("T", "V") for x in read_smi(AVE / AVE_DIR[t] / ("inactive_%s.smi" % k))})
    rng = random.Random(LS["litpcba"]["seed"] + i)
    n_before = len(act)
    if len(act) > LS["litpcba"]["actives_cap"]:
        act = rng.sample(act, LS["litpcba"]["actives_cap"])
    pick = rng.sample(ina, NEG * len(act))
    for lid, smi in sorted(act):
        add("litpcba", "litpcba_" + t, "active", 0, lid, smi, "actives_total=%d" % n_before)
    for neg, k in split_rounds(pick, len(act)):
        for lid, smi in neg:
            add("litpcba", "litpcba_" + t, "negative", k, lid, smi, "inactive")
    LOG["sets"]["litpcba/" + t] = dict(n_active_total=n_before, n_active=len(act), n_negative=len(pick), n_inactives_available=len(ina))
zrows = list(csv.DictReader(open(ROOT / "inputs/external/zinc20/zinc_pool.tsv", encoding="utf-8"), delimiter="\t"))
zrng = random.Random(LS["zinc_unmatched"]["seed"])
order = list(range(len(zrows))); zrng.shuffle(order)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return gen.GetFingerprint(m) if m is not None else None


act_fps = {t: [f for f in (fp(s) for _, s in dude_act[t]) if f is not None] for t in DT}
t0 = time.time()
cand_idx = order[:NLABEL]
with Pool(NPROC) as pool:
    lab = pool.map(PR.cheap_props, [zrows[i]["smiles"] for i in cand_idx], chunksize=64)
    act_lab = {t: pool.map(PR.cheap_props, [s for _, s in dude_act[t]], chunksize=16) for t in DT}
zcls = {i: klass(r["dominant_charge"]) for i, r in zip(cand_idx, lab)}
LOG["zinc_labelled"] = dict(n=len(cand_idx), n_ok=sum(1 for r in lab if r["ok"]), class_counts=dict(Counter(v for v in zcls.values() if v)), seconds=round(time.time() - t0, 1))
zfp = {}


def zfingerprint(i):
    if i not in zfp:
        zfp[i] = fp(zrows[i]["smiles"])
    return zfp[i]


def max_sim(i, t):
    f = zfingerprint(i)
    return max(DataStructs.BulkTanimotoSimilarity(f, act_fps[t])) if f is not None else 1.0


thr = LS["zinc_unmatched"]["exclude_ecfp4_tanimoto_ge"]
used_unmatched = defaultdict(set)
used_any = set()
dist_log = {}
trng = random.Random(LS["zinc_unmatched"]["seed"] + 1)
tlist = list(range(len(DT))); trng.shuffle(tlist)
for ti in tlist:
    t = DT[ti]; n_act = len(dude_act[t]); need = NEG * n_act
    chosen, rej = [], 0
    for i in order:
        if i in used_any:
            continue
        s = max_sim(i, t)
        if s >= thr:
            rej += 1; continue
        chosen.append((i, s)); used_any.add(i); used_unmatched[t].add(i)
        if len(chosen) == need:
            break
    assert len(chosen) == need, (t, len(chosen), need)
    trng.shuffle(chosen)
    for neg, k in split_rounds(chosen, n_act):
        for i, s in neg:
            add("zinc_unmatched", t, "negative", k, zrows[i]["zinc_id"], zrows[i]["smiles"], "maxsim=%.3f" % s)
    LOG["sets"]["zinc_unmatched/" + t] = dict(n_negative=need, n_rejected_by_similarity=rej)
mrng = random.Random(LS["zinc_charge_mismatch"]["seed"])
used_mm = set()
lab_by_class = {c: [i for i in cand_idx if zcls.get(i) == c] for c in CLASSES}
for ti in tlist:
    t = DT[ti]; n_act = len(dude_act[t]); need = NEG * n_act
    ac = Counter(klass(r["dominant_charge"]) for r in act_lab[t] if r["ok"] and r["dominant_charge"] is not None)
    tot = sum(ac.values())
    p = {c: ac.get(c, 0) / tot for c in CLASSES}
    w = {c: 1.0 - p[c] for c in CLASSES}; sw = sum(w.values()); w = {c: w[c] / sw for c in CLASSES}
    raw = {c: w[c] * need for c in CLASSES}
    quota = {c: int(raw[c]) for c in CLASSES}
    for c in sorted(CLASSES, key=lambda c: raw[c] - quota[c], reverse=True)[: need - sum(quota.values())]:
        quota[c] += 1
    chosen, short = [], {}
    for c in CLASSES:
        got = []
        for i in lab_by_class[c]:
            if len(got) == quota[c]:
                break
            if i in used_mm or i in used_unmatched[t]:
                continue
            s = max_sim(i, t)
            if s >= thr:
                continue
            got.append((i, s, c))
        for i, s, c_ in got:
            used_mm.add(i)
        short[c] = quota[c] - len(got)
        chosen += got
    if sum(short.values()):
        for c in sorted(CLASSES, key=lambda c: w[c], reverse=True):
            for i in lab_by_class[c]:
                if len(chosen) == need:
                    break
                if i in used_mm or i in used_unmatched[t]:
                    continue
                s = max_sim(i, t)
                if s >= thr:
                    continue
                chosen.append((i, s, c)); used_mm.add(i)
    assert len(chosen) == need, (t, len(chosen), need)
    mrng.shuffle(chosen)
    for neg, k in split_rounds(chosen, n_act):
        for i, s, c in neg:
            add("zinc_charge_mismatch", t, "negative", k, zrows[i]["zinc_id"], zrows[i]["smiles"], "class=%s;maxsim=%.3f" % (c, s))
    dist_log[t] = dict(active_class_fraction={c: round(p[c], 3) for c in CLASSES}, weights={c: round(w[c], 3) for c in CLASSES}, quota=quota, shortfall=short,
                       drawn_class_counts=dict(Counter(c for _, _, c in chosen)), n_active_classified=tot)
    LOG["sets"]["zinc_charge_mismatch/" + t] = dict(n_negative=need)
LOG["zinc_charge_mismatch_distribution"] = dist_log

orng = random.Random(20261023)
tkeys = sorted({(r["set"], r["target"]) for r in rows}); orng.shuffle(tkeys)
rank = {k: i for i, k in enumerate(tkeys)}
rows.sort(key=lambda r: (r["round"], rank[(r["set"], r["target"])], r["ligand_id"]))
COLS = [("set", "", "dude_full / litpcba / zinc_unmatched / zinc_charge_mismatch (the three DUD-E-receptor sets share the actives: actives appear once, under dude_full)"), ("target", "", "receptor target (litpcba_* for LIT-PCBA)"),
        ("role", "", "active / negative"), ("round", "", "0 = actives, k = k-th third of the negatives of the target"), ("ligand_id", "", "ChEMBL id / ZINC id / LIT-PCBA id"), ("smiles", "", "SMILES from the source file"),
        ("info", "", "decoy / inactive; ZINC sets: class (dominant-charge class) and max ECFP4 Tanimoto to the actives of the target")]
out = QDIR / "ligand_sets.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w_ = csv.writer(f); w_.writerow([c[0] for c in COLS])
    for r in rows:
        w_.writerow([r[c[0]] for c in COLS])
cnt = Counter((r["set"], r["role"]) for r in rows)
LOG["counts"] = {"%s/%s" % k: v for k, v in sorted(cnt.items())}
LOG["n_rows"] = len(rows)
BLOCK = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching.yaml", [ROOT / "inputs/external/zinc20/zinc_pool.tsv"], seed=LS["zinc_unmatched"]["seed"], extra=dict(nlabel=NLABEL))
P.write_schema(out, COLS, "one row per (set, target, ligand) to be docked", "Ligand sets and docking order of decoy-matching", extra=dict(provenance=BLOCK))
LOG["provenance"] = BLOCK
json.dump(LOG, open(RES / "set_build_log.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: LOG[k] for k in ("counts", "n_rows", "zinc_labelled")}, indent=1))
