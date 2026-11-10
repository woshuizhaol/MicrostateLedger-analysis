import hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import casf_common as C

WORK = C.RAW / "scoring_work_states"
os.environ["CASF_WORK"] = str(WORK)
C.RAW.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
import casf_scoring_protocol as S
import charge_common as C3

C.check_pins()
main, ad4, aff = C.load_d11()
PIDS = sorted(main)
PRED = C.CFG["predictors"]["list"]
NAGREE = C.CFG["predictors"]["consensus_min_agree"]
NAMED = ["input", "neutral", "single", "dominant", "random"]


def info(pid):
    wd = tempfile.mkdtemp(prefix="st_%s_" % pid, dir=str(WORK))
    try:
        j = main[pid]
        native_noH, smi, _ = S.load_native(pid, wd)
        assert smi == j["input_smiles"], "input SMILES changed for " + pid
        states, ens, trunc = S.enumerate_states(smi, native_noH)
        r = C3.O.analyze_compound(smi, C3.TABLE)
        pop = {x["smiles"]: x["population"] for x in r["rows"]} if r else {}
        return pid, dict(ens=ens, trunc=trunc, states=states, pop=pop,
                         q_popmax=C3.O.fcharge(r["pop_argmax_smiles"]) if r else None, popmax=r["pop_argmax_smiles"] if r else None,
                         q_dom_old=r["dom_charge"] if r else None, single_old=r["single_smiles"] if r else None)
    finally:
        shutil.rmtree(wd, ignore_errors=True)


t0 = time.time()
with ProcessPoolExecutor(max_workers=C.CFG["validation"]["nproc"]) as ex:
    INFO = dict(ex.map(info, PIDS))
C.log("re-enumerated ensembles and HH populations for", len(INFO), "complexes in %.0fs" % (time.time() - t0))

chk = dict(ensemble_size_equal=0, truncated_equal=0, ens_subset_of_scored=0, named_states_equal=0, named_scores_equal=0, named_scores_checked=0, analyze_failed=0, random_stored_in_pool=0,
           dominant_charge_equal_old=0, single_equal_old=0)
for pid in PIDS:
    j, I = main[pid], INFO[pid]
    chk["ensemble_size_equal"] += len(I["ens"]) == j["ensemble_size"]
    chk["truncated_equal"] += I["trunc"] == j["ensemble_truncated"]
    chk["ens_subset_of_scored"] += set(I["ens"]) <= set(j["all_scored"])
    chk["named_states_equal"] += all((I["states"][n] or None) == (j["conditions"][n].get("smiles") or None) for n in NAMED if n != "random")
    pool_nd = [s_ for s_ in I["ens"] if s_ != I["states"]["single"]]
    sr = j["conditions"]["random"].get("smiles") or None
    chk["random_stored_in_pool"] += (sr in pool_nd) if pool_nd else (sr is None)
    chk["analyze_failed"] += not I["pop"]
    if I["pop"]:
        chk["dominant_charge_equal_old"] += I["q_dom_old"] == main[pid]["conditions"]["dominant"]["q"]
        chk["single_equal_old"] += I["single_old"] == j["conditions"]["single"]["smiles"]
    for n in NAMED:
        s = j["conditions"][n].get("smiles")
        if not s:
            continue
        for sc in C.SCORERS:
            a, b = C.stored_cond_score(main, ad4, pid, n, sc), C.member_score(main, ad4, pid, s, sc)
            chk["named_scores_checked"] += 1
            chk["named_scores_equal"] += (a == b) or (not C.fin(a) and not C.fin(b))
assert chk["ensemble_size_equal"] == chk["truncated_equal"] == chk["ens_subset_of_scored"] == chk["named_states_equal"] == chk["random_stored_in_pool"] == len(PIDS), chk
assert chk["named_scores_equal"] == chk["named_scores_checked"], chk

IN = C.CFG["predictors"]["input_for_prediction"]
assert IN == "input_smiles", IN
inp = [main[p][IN] for p in PIDS]
ob = C3.openbabel_batch(inp)
fin_ml, fout_ml = C.RAW / "ml_in_casf.tsv", C.RAW / "ml_sites_casf.jsonl"
uniq = list(dict.fromkeys(inp))
fin_ml.write_text("smiles\n" + "\n".join(uniq) + "\n", encoding="utf-8")
pr = subprocess.run([str(C.PATHS["ml_python"]), "-B", str(C.PATHS["ml_wrapper"]), str(fin_ml), str(fout_ml), str(C.CFG["validation"]["nproc"])], capture_output=True, text=True, cwd=str(C.ROOT))
if pr.returncode != 0:
    sys.exit("MolGpKa wrapper failed: " + pr.stderr[-400:])
C.log("ml:", pr.stdout.strip().splitlines()[-1])
ml = {r["smiles"]: r for r in (json.loads(l) for l in open(fout_ml, encoding="utf-8"))}

ROWS, member_q = [], {}
for pid in PIDS:
    j, a, I = main[pid], ad4[pid], INFO[pid]
    roles = {}
    for n in NAMED:
        s = j["conditions"][n].get("smiles")
        if s:
            roles.setdefault(s, []).append(n)
    for s in I["ens"]:
        roles.setdefault(s, []).append("ensemble")
    member_q[pid] = {}
    for i, (s, e) in enumerate(j["all_scored"].items()):
        member_q[pid][s] = e["q"]
        p = I["pop"].get(s)
        ROWS.append(dict(pdbid=pid, microstate_id="%s_m%02d" % (pid, i), smiles=s, q=e["q"], vina=e.get("vina"), smina=e.get("smina"), ad4=a["all_ad4"].get(s),
                         roles=";".join(roles.get(s, [])), in_ensemble=s in I["ens"], hh_population=p if p is not None and not math.isnan(p) else None,
                         mapped=e.get("mapped"), pdbqt_via=e.get("pdbqt_via")))
COLS = [("pdbid", "", "CASF-2016 complex"), ("microstate_id", "", "pdbid_mNN, NN = position in the stored all_scored dictionary of D11"), ("smiles", "", "microstate SMILES (canonical, as stored)"),
        ("q", "e", "formal net charge"), ("vina", "kcal/mol", "in-place Vina score_only on the native pose (empty = not scored)"), ("smina", "kcal/mol", "in-place smina score_only, Vina force field (empty = not scored)"),
        ("ad4", "kcal/mol", "in-place smina --scoring ad4_scoring, Gasteiger-charged receptor (empty = not scored)"),
        ("roles", "", "semicolon list: input, neutral, single (default state), dominant (Dimorphite), random (the single original random draw), ensemble (member of the 6.4-8.4 Dimorphite ensemble, truncated to 24)"),
        ("in_ensemble", "", "member of the ensemble (see roles)"), ("hh_population", "", "independent-site HH population among the untruncated ensemble members (original analyze_compound); empty when not an ensemble member"),
        ("mapped", "", "microstate could be mapped onto the native heavy-atom coordinates"), ("pdbqt_via", "", "route used to write the ligand pdbqt (meeko or obabel)")]
BLOCK = C.P.provenance_block(Path(__file__), C.ROOT / "configs/casf_scoring.yaml", [C.PATHS["d11_summary"], C.PATHS["d11_per_complex"], C.PATHS["original_scoring_py"], C.PATHS["original_ad4_py"], C.PATHS["coreset_dat"],
                             fout_ml], seed=None, extra=dict(d11_score_digest=C.CFG["d11_score_digest"], obabel_version=subprocess.run([str(C.PATHS["obabel"]), "-V"], capture_output=True, text=True).stdout.strip()),
                             tools_extra=("rdkit", "meeko"))
out1 = C.RAW / "per_microstate_inplace_scores.csv"
C.write_csv(out1, COLS, ROWS)
C.P.write_schema(out1, COLS, "per-microstate (every microstate stored by the original CASF-2016 analysis)", "In-place (native-pose) Vina / smina / AD4 scores of every stored microstate of the 285 CASF-2016 complexes", extra=dict(provenance=BLOCK))

QCOLS_ROWS, PRED_ROWS = [], []
for pid in PIDS:
    j, I = main[pid], INFO[pid]
    single = j["conditions"]["single"]["smiles"]; dom = j["conditions"]["dominant"]["smiles"]
    cand = list(dict.fromkeys(I["ens"] + ([dom] if dom else [])))
    obq = ob.get(PIDS.index(pid))
    mls = C3.ml_state(ml.get(j[IN]))
    qs = dict(dimorphite_dom=j["conditions"]["dominant"]["q"], openbabel=obq[0] if obq and obq[2] else None, enum_popmax=I["q_popmax"], ml_pka=mls[0] if mls else None)
    cnt = Counter(v for v in qs.values() if v is not None)
    cons = [q for q, n in cnt.items() if n >= NAGREE]
    qs["consensus_ge3"] = cons[0] if cons else None
    for p in PRED:
        qp = qs[p]
        if p == "dimorphite_dom":
            chosen, status = dom, "direct"
        elif qp is None:
            chosen, status = None, "no_consensus" if p == "consensus_ge3" else "no_prediction"
        else:
            pool = [s for s in cand if member_q[pid][s] == qp]
            if pool:
                chosen = sorted(pool, key=lambda s: (-(I["pop"].get(s) if I["pop"].get(s) is not None and not math.isnan(I["pop"].get(s)) else -1.0), s))[0]
                status = "chosen"
            else:
                chosen, status = None, "no_candidate_with_charge"
        if chosen is None:
            chosen_state, final = single, "fallback_single"
        else:
            chosen_state, final = chosen, status
        PRED_ROWS.append(dict(pdbid=pid, predictor=p, q_pred=qp, q_single=member_q[pid][single], chosen_smiles=chosen_state, chosen_q=member_q[pid][chosen_state], status=status,
                              fallback=chosen is None, differs_from_single=chosen_state != single, charge_differs_from_single=member_q[pid][chosen_state] != member_q[pid][single],
                              n_candidates=len(cand), n_candidates_with_pred_charge=sum(1 for s in cand if qp is not None and member_q[pid][s] == qp)))
PC = [("pdbid", "", "CASF-2016 complex"), ("predictor", "", "dimorphite_dom, openbabel, enum_popmax, ml_pka, consensus_ge3"), ("q_pred", "e", "net charge predicted for the input ligand (empty = no prediction)"),
      ("q_single", "e", "net charge of the default (single) state"), ("chosen_smiles", "", "microstate scored for this predictor (the default state when fallback)"), ("chosen_q", "e", "net charge of that microstate"),
      ("status", "", "direct (Dimorphite dominant, as in the original analysis) / chosen (candidate with the predicted charge) / no_prediction / no_consensus / no_candidate_with_charge"),
      ("fallback", "", "True when the default state is kept because there is no usable prediction or no candidate with the predicted charge"),
      ("differs_from_single", "", "chosen microstate is not the default state (structure)"), ("charge_differs_from_single", "", "chosen microstate has a different net charge from the default state"),
      ("n_candidates", "", "ensemble members + Dimorphite dominant"), ("n_candidates_with_pred_charge", "", "candidates having the predicted net charge")]
out2 = C.RAW / "predictor_states.csv"
C.write_csv(out2, PC, PRED_ROWS)
C.P.write_schema(out2, PC, "complex x predictor (285 x 5)", "Net charge predicted by each charge-state predictor for each CASF-2016 ligand and the microstate scored for it", extra=dict(provenance=BLOCK))

summ = dict(provenance=BLOCK, n_complexes=len(PIDS), n_microstates=len(ROWS), checks=chk, per_predictor={})
for p in PRED:
    rr = [r for r in PRED_ROWS if r["predictor"] == p]
    summ["per_predictor"][p] = dict(status=dict(Counter(r["status"] for r in rr)), n_differs_from_single=sum(r["differs_from_single"] for r in rr),
                                    n_charge_differs=sum(r["charge_differs_from_single"] for r in rr))
summ["pairwise_charge_agreement_with_dimorphite"] = {p: sum(1 for a, b in zip([r for r in PRED_ROWS if r["predictor"] == "dimorphite_dom"], [r for r in PRED_ROWS if r["predictor"] == p])
                                                           if a["q_pred"] is not None and a["q_pred"] == b["q_pred"]) for p in PRED}
C.RES.mkdir(parents=True, exist_ok=True)
json.dump(summ, open(C.RES / "states_summary.json", "w"), indent=2, ensure_ascii=False)
C.log("microstate rows", len(ROWS), "| checks", json.dumps(chk))
for p in PRED:
    C.log(p, json.dumps(summ["per_predictor"][p]))
shutil.rmtree(WORK, ignore_errors=True)
