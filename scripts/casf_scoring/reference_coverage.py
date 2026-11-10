import csv, json, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import casf_common as C
import charge_common as C3
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS

RDLogger.DisableLog("rdApp.*")
C.check_pins()
AN = C.CFG["anchoring"]
t0 = time.time()
main, ad4, aff = C.load_d11()
PIDS = sorted(main)
PRED = C.CFG["predictors"]["list"]
PS = {(r["pdbid"], r["predictor"]): r for r in csv.DictReader(open(C.RAW / "predictor_states.csv", encoding="utf-8"))}


def key_of(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return C3.skeleton_key(m)


LK = {p: key_of(main[p]["input_smiles"]) for p in PIDS}
assert all(LK.values())
A = list(csv.DictReader(open(C.PATHS["anchor_set"], encoding="utf-8")))
A_by = defaultdict(list)
for r in A:
    A_by[r["ik_skeleton"]].append(r)
iu = list(csv.DictReader(open(C.PATHS["iupac_csv"], encoding="utf-8")))
ch = list(csv.DictReader(open(C.E(C.CFG["paths"].get("chembl_tsv", "$ANALYSIS_ROOT/inputs/external/chembl/chembl_pka_activities.tsv")), encoding="utf-8"), delimiter="\t"))
uniq = sorted({r["SMILES"] for r in iu if r["SMILES"]} | {r["canonical_smiles"] for r in ch if r["canonical_smiles"]})
with ProcessPoolExecutor(max_workers=C.CFG["validation"]["nproc"]) as ex:
    keys = list(ex.map(key_of, uniq, chunksize=200))
KEY = dict(zip(uniq, keys))
B_by = {"IUPAC": defaultdict(list), "ChEMBL": defaultdict(list)}
for r in iu:
    k = KEY.get(r["SMILES"])
    if k:
        B_by["IUPAC"][k].append(r)
for r in ch:
    k = KEY.get(r["canonical_smiles"])
    if k:
        B_by["ChEMBL"][k].append(r)
C.log("layer B keys: IUPAC", len(B_by["IUPAC"]), "ChEMBL", len(B_by["ChEMBL"]), "%.0fs" % (time.time() - t0))


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


A_mols = {}
for r in A:
    m = Chem.MolFromSmiles(r["smiles"])
    if m is not None:
        A_mols[r["mol_id"]] = (m, fp(m))


def analogue(pid):
    m = Chem.MolFromSmiles(main[pid]["input_smiles"]); big = C3.clean_mol(m)
    f = fp(big)
    sims = sorted(((DataStructs.TanimotoSimilarity(f, v[1]), k) for k, v in A_mols.items()), reverse=True)[:10]
    best = (0.0, None, 0.0)
    for s, k in sims:
        if s < 0.4:
            break
        a = A_mols[k][0]
        res = rdFMCS.FindMCS([big, a], timeout=2, ringMatchesRingOnly=True, completeRingsOnly=False)
        cov = res.numAtoms / max(1, min(big.GetNumHeavyAtoms(), a.GetNumHeavyAtoms()))
        if cov > best[0]:
            best = (cov, k, s)
    return pid, best


with ProcessPoolExecutor(max_workers=C.CFG["validation"]["nproc"]) as ex:
    AN_BEST = dict(ex.map(analogue, PIDS))
C.log("analogues done %.0fs" % (time.time() - t0))

rows, subset_rows = [], []
cov = dict(n_ligands=len(PIDS), match_key=AN["match_key"], min_n_for_inference=AN["min_n_for_inference"], mcs_similarity_min=AN["mcs_similarity_min"])
matA = [p for p in PIDS if LK[p] in A_by]
matB = {s: [p for p in PIDS if LK[p] in B_by[s]] for s in B_by}
matB_any = [p for p in PIDS if any(LK[p] in B_by[s] for s in B_by)]
completeA = [p for p in matA if any(r["complete_profile"] == "True" for r in A_by[LK[p]])]
relaxedA = [p for p in matA if any(r["complete_relaxed"] == "True" for r in A_by[LK[p]])]
analog = [p for p in PIDS if AN_BEST[p][0] >= AN["mcs_similarity_min"]]
cov.update(layerA_exact_anchor_set=len(matA), layerA_complete_strict=len(completeA), layerA_complete_relaxed=len(relaxedA), layerA_with_exp_dominant_charge=len([p for p in matA if A_by[LK[p]][0]["exp_dominant_charge"] != ""]),
           layerB_exact_any_record=len(matB_any), layerB_by_source={s: len(v) for s, v in matB.items()}, layerB_not_in_anchor_set=len([p for p in matB_any if p not in matA]),
           layerC_close_analogue_mcs=len(analog), layerC_close_analogue_among_unmatched=len([p for p in analog if p not in matB_any]),
           fraction_exact_any_record=len(matB_any) / len(PIDS), fraction_complete_strict=len(completeA) / len(PIDS))
inference = len(completeA) >= AN["min_n_for_inference"]
cov["inference_allowed"] = inference
cov["matched_complexes_anchor_set"] = matA; cov["matched_complexes_any_record"] = matB_any; cov["complete_strict_complexes"] = completeA
cov["unmatched_reason_note"] = "charge-state anchor set keeps neutral, drug-like (MW/rings/HBD/HBA), element-restricted molecules with quality-filtered constants; layer B has no filter."

for p in matA:
    ar = A_by[LK[p]]
    a0 = next((r for r in ar if r["complete_profile"] == "True"), ar[0])
    exp_q = int(a0["exp_dominant_charge"]) if a0["exp_dominant_charge"] != "" else None
    for pr in PRED:
        ps = PS[(p, pr)]
        qp = int(ps["q_pred"]) if ps["q_pred"] != "" else None
        rows.append(dict(pdbid=p, ligand_smiles=main[p]["input_smiles"], ik_skeleton=LK[p], anchor_mol_id=a0["mol_id"], anchor_source=a0["source"], n_anchor_records=len(ar), complete_profile=a0["complete_profile"] == "True",
                         complete_relaxed=a0["complete_relaxed"] == "True", n_sites_pred=a0["n_sites_pred"], n_pka_exp_window=a0["n_pka_exp_window"], ambiguous=a0["ambiguous"] == "True", coupled_sites=a0["coupled_sites"] == "True",
                         exp_dominant_charge=exp_q, q_single=int(ps["q_single"]), predictor=pr, q_pred=qp, status=ps["status"], agrees_with_exp=(qp == exp_q) if (qp is not None and exp_q is not None) else None,
                         single_agrees_with_exp=(int(ps["q_single"]) == exp_q) if exp_q is not None else None))
agree = {}
for pr in PRED + ["single"]:
    for nm, sel in (("matched_with_exp_dominant", [r for r in rows if r["exp_dominant_charge"] is not None]), ("complete_strict", [r for r in rows if r["complete_profile"] and r["exp_dominant_charge"] is not None])):
        if pr == "single":
            rr = {r["pdbid"]: r["single_agrees_with_exp"] for r in sel}
            agree.setdefault(nm, {})[pr] = dict(k=sum(1 for v in rr.values() if v), n=len(rr))
        else:
            rr = [r for r in sel if r["predictor"] == pr and r["agrees_with_exp"] is not None]
            agree.setdefault(nm, {})[pr] = dict(k=sum(1 for r in rr if r["agrees_with_exp"]), n=len(rr))
cov["agreement_with_experimental_dominant_charge"] = agree
cov["note"] = ("n is far below min_n_for_inference: coverage and descriptive agreement only; no Rp / dRp analysis on the experimentally defined state." if not inference
               else "inference subset written to exp_anchor_subset.csv")
cov["analogue_table_head"] = {p: dict(mcs_cover=round(AN_BEST[p][0], 3), tanimoto=round(AN_BEST[p][2], 3), anchor_mol_id=AN_BEST[p][1]) for p in analog[:15]}

MC = [("pdbid", "", "CASF-2016 complex"), ("ligand_smiles", "", "input SMILES of the ligand (D11)"), ("ik_skeleton", "", "first InChIKey block of the cleaned ligand"), ("anchor_mol_id", "", "charge-state anchor-set molecule id (the complete-profile record when several)"),
      ("anchor_source", "", "IUPAC / ChEMBL / SAMPL source taken for that molecule"), ("n_anchor_records", "", "anchor-set rows sharing the skeleton"), ("complete_profile", "", "charge-state strict completeness gate: experimental constants cover every predicted ionisable group in the window"),
      ("complete_relaxed", "", "charge-state relaxed gate"), ("n_sites_pred", "", "predicted ionisable groups in the gate window (charge-state)"), ("n_pka_exp_window", "", "experimental constants inside the window"), ("ambiguous", "", "charge-state flag"),
      ("coupled_sites", "", "charge-state flag"), ("exp_dominant_charge", "e", "experimental pH-7.4 dominant net charge from the merged ladder (independent-site HH); reliable only when complete_profile is True"),
      ("q_single", "e", "net charge of the default (single) state"), ("predictor", "", "predictor"), ("q_pred", "e", "net charge predicted"), ("status", "", "state selection status"),
      ("agrees_with_exp", "", "q_pred == exp_dominant_charge"), ("single_agrees_with_exp", "", "default state net charge == exp_dominant_charge")]
BLOCK = C.P.provenance_block(Path(__file__), C.ROOT / "configs/casf_scoring.yaml", [C.PATHS["anchor_set"], C.PATHS["iupac_csv"], C.RAW / "predictor_states.csv"], seed=None, extra=dict(n_ligands=len(PIDS)), tools_extra=("rdkit",))
cov["provenance"] = BLOCK
C.RES.mkdir(parents=True, exist_ok=True)
out = C.RES / "exp_anchor_matches.csv"
C.write_csv(out, MC, rows)
C.P.write_schema(out, MC, "matched CASF ligand x predictor", "CASF-2016 ligands that match a charge-state anchor-set molecule (exact cleaned InChIKey skeleton) and what each predictor says about them", extra=dict(provenance=BLOCK))
sub = C.RES / "exp_anchor_subset.csv"
if inference:
    sub_rows = [r for r in rows if r["complete_profile"]]
    C.write_csv(sub, MC, sub_rows)
    C.P.write_schema(sub, MC, "complete-profile matched ligand x predictor", "Inference subset (n >= min_n_for_inference)", extra=dict(provenance=BLOCK))
elif sub.exists():
    sub.unlink()
json.dump(cov, open(C.RES / "exp_anchor_coverage.json", "w"), indent=2, ensure_ascii=False)
C.log("coverage:", json.dumps({k: v for k, v in cov.items() if k.startswith(("layer", "inference", "fraction"))}))
C.log("agreement:", json.dumps(agree))
