import json, sys, time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import casf_common as C

C.check_pins()
ST, PRED = C.CFG["stats"], C.CFG["predictors"]["list"]
B, NB = ST["n_null"], ST["n_boot"]
assert set(ST["null_variants"]) == {"non_single", "incl_single"}, ST["null_variants"]
t0 = time.time()
PIDS, AFF, M, PS, COMMON = C.load_tables()
main, ad4, _ = C.load_d11()
POS = {p: i for i, p in enumerate(PIDS)}
U = C.uniform_matrix(len(PIDS), B, ST["seed_null"])
C.log("uniform matrix", U.shape, "%.1fs" % (time.time() - t0))
TOL = C.CFG["affected_subset"]["score_change_tol"]


def member(pid, smi):
    return next(m for m in M[pid] if m["smiles"] == smi)


def role(pid, r):
    return next(m for m in M[pid] if r in m["roles"])


def fin_scores(ms, sc):
    return np.array([m["score"][sc] for m in ms if C.fin(m["score"][sc])], float)


fail = []
OUTCOLS_NULL = [("b", "", "draw number, seed = seed_null + b"), ("Rp", "", "Pearson Rp of the drawn states over n_common complexes"), ("dRp", "", "Rp(drawn) - Rp(single)"),
                ("RMSE", "", "OLS RMSE of the drawn states"), ("dRMSE", "", "RMSE(drawn) - RMSE(single)")]
BLOCK = C.P.provenance_block(Path(__file__), C.ROOT / "configs/casf_scoring.yaml", [C.RAW / "per_microstate_inplace_scores.csv", C.RAW / "predictor_states.csv", C.PATHS["coreset_dat"]], seed=ST["seed_null"], n_boot=NB,
                             extra=dict(seed_boot=ST["seed_boot"], n_null=B, null_pool_rule=ST["null_pool_rule"]))
SUMROWS, ROB, AFFROWS, PCROWS, SUMM = [], [], [], [], dict(provenance=BLOCK, scorers={})


def write_null(name, sc, Rp, dRp, RM_, dRM, desc):
    rows = [dict(b=b + 1, Rp=Rp[b], dRp=dRp[b], RMSE=RM_[b], dRMSE=dRM[b]) for b in range(B)]
    path = C.RES / name
    C.write_csv(path, OUTCOLS_NULL, rows)
    C.P.write_schema(path, OUTCOLS_NULL, "one row per random draw (%d)" % B, desc, extra=dict(provenance=BLOCK))


for sc in C.SCORERS:
    cl = COMMON[sc]; n = len(cl)
    if n != C.CFG["n_common_expected"][sc]:
        fail.append("n_common %s = %d" % (sc, n))
    cols = np.array([POS[p] for p in cl])
    y = np.array([AFF[p] for p in cl])
    sing = [role(p, "single") for p in cl]
    s_single = np.array([m["score"][sc] for m in sing], float)
    assert np.isfinite(s_single).all()
    for p, m in zip(cl, sing):
        assert C.stored_cond_score(main, ad4, p, "single", sc) == m["score"][sc]
    Rp_s, RM_s = C.rp_rmse(y, s_single)
    s_dom_old = np.array([C.stored_cond_score(main, ad4, p, "dominant", sc) for p in cl], float)
    s_rand_old = np.array([C.stored_cond_score(main, ad4, p, "random", sc) for p in cl], float)
    d_dom_old = C.rp_rmse(y, s_dom_old)[0] - Rp_s
    d_rand_old = C.rp_rmse(y, s_rand_old)[0] - Rp_s
    if round(d_dom_old, 4) != C.CFG["original_delta_expected"][sc]:
        fail.append("original dominant delta %s %.4f" % (sc, d_dom_old))
    if round(d_rand_old, 4) != C.CFG["original_random_delta_expected"][sc]:
        fail.append("original random delta %s %.4f" % (sc, d_rand_old))
    pool_ns = [fin_scores([m for m in M[p] if m["in_ens"] and "single" not in m["roles"]], sc) for p in cl]
    pool_in = [fin_scores([m for m in M[p] if m["in_ens"]], sc) for p in cl]
    n_empty = sum(1 for q in pool_ns if len(q) == 0)
    if n_empty:
        fail.append("empty non-single pool in common set %s: %d" % (sc, n_empty))
    sp, ch_state, ch_charge, fb = {}, {}, {}, {}
    for pr in PRED:
        rows = [PS[(p, pr)] for p in cl]
        ms = [member(p, r["chosen_smiles"]) for p, r in zip(cl, rows)]
        assert all(C.fin(m["score"][sc]) for m in ms), (sc, pr)
        sp[pr] = np.array([m["score"][sc] for m in ms], float)
        ch_state[pr] = np.array([r["differs_from_single"] == "True" for r in rows])
        ch_charge[pr] = np.array([r["charge_differs_from_single"] == "True" for r in rows])
        fb[pr] = np.array([r["fallback"] == "True" for r in rows])
    idx = C.boot_indices(n, NB, ST["seed_boot"])
    Ucl = U[:, cols]
    r_ns, m_ns = C.null_rows(y, s_single, pool_ns, Ucl)
    r_in, m_in = C.null_rows(y, s_single, pool_in, Ucl)
    d_ns, d_in = r_ns - Rp_s, r_in - Rp_s
    dm_ns, dm_in = m_ns - RM_s, m_in - RM_s
    write_null("random_null_%s.csv" % sc, sc, r_ns, d_ns, m_ns, dm_ns, "Random microstate null, non-single pool: every complex gets one random ensemble member other than the default state (%s, n_common=%d)" % (sc, n))
    write_null("random_null_incl_single_%s.csv" % sc, sc, r_in, d_in, m_in, dm_in, "Random microstate null, pool includes the default state (%s, n_common=%d)" % (sc, n))
    SUMM["scorers"][sc] = dict(n_common=n, Rp_single=Rp_s, RMSE_single=RM_s, original_dominant_dRp=d_dom_old, original_random_dRp=d_rand_old, n_empty_nonsingle_pool=n_empty,
                               original_random_percentile_in_null=float(np.mean(d_ns <= d_rand_old)), n_pool_size_1=int(sum(1 for q in pool_ns if len(q) == 1)),
                               pool_size_median=float(np.median([len(q) for q in pool_ns])), null_ns_mean=float(d_ns.mean()), null_ns_sd=float(d_ns.std(ddof=1)),
                               frac_null_dRp_gt0=float(np.mean(d_ns > 0)))
    for pr in PRED:
        s_p = sp[pr]
        Rp_p, RM_p = C.rp_rmse(y, s_p)
        obs, obs_m = Rp_p - Rp_s, RM_p - RM_s
        bR, bM = C.paired_delta_boot(y, s_single, s_p, idx)
        lo, hi = C.pct(bR); mlo, mhi = C.pct(bM)
        pools_ch = [pool_ns[i] if ch_state[pr][i] else np.array([]) for i in range(n)]
        r_ch, m_ch = C.null_rows(y, s_single, pools_ch, Ucl)
        d_ch = r_ch - Rp_s
        pools_mt, keep_mt = [], np.zeros(n, bool)
        for i, p in enumerate(cl):
            if not ch_state[pr][i]:
                pools_mt.append(np.array([])); continue
            q0 = sing[i]["q"]; chosen = PS[(p, pr)]["chosen_smiles"]; dq = abs(member(p, chosen)["q"] - q0)
            cand = [m for m in M[p] if (m["in_ens"] or "dominant" in m["roles"]) and "single" not in m["roles"] and m["smiles"] != chosen and abs(m["q"] - q0) == dq]
            pl = fin_scores(cand, sc); pools_mt.append(pl); keep_mt[i] = len(pl) > 0
        s_obs_mt = np.where(keep_mt, s_p, s_single)
        obs_mt = C.rp_rmse(y, s_obs_mt)[0] - Rp_s
        r_mt, m_mt = C.null_rows(y, s_single, pools_mt, Ucl)
        d_mt = r_mt - Rp_s
        if pr == "dimorphite_dom":
            write_null("changed_set_null_dimorphite_dom_%s.csv" % sc, sc, r_ch, d_ch, m_ch, m_ch - RM_s, "Random microstate null restricted to the complexes where the Dimorphite dominant state differs from the default (%s)" % sc)
            write_null("matched_null_dimorphite_dom_%s.csv" % sc, sc, r_mt, d_mt, m_mt, m_mt - RM_s, "Matched-|dq| random null for the Dimorphite dominant state (%s)" % sc)
        for nt, dn, ob, npert in (("non_single", d_ns, obs, n), ("incl_single", d_in, obs, n), ("changed_set", d_ch, obs, int(ch_state[pr].sum())), ("matched_dq", d_mt, obs_mt, int(keep_mt.sum()))):
            s_ = C.null_summary(dn, ob)
            SUMROWS.append(dict(scorer=sc, null_type=nt, predictor=pr, n_perturbed=npert, n_draws=B, null_mean=s_["null_mean"], q025=s_["q025"], q975=s_["q975"], obs_dRp=s_["obs_dRp"], p_emp=s_["p_emp"],
                                obs_within_null_95=s_["obs_within_null_95"], frac_null_ge_obs=s_["frac_null_ge_obs"]))
        pn = {r["null_type"]: r["p_emp"] for r in SUMROWS if r["scorer"] == sc and r["predictor"] == pr}
        ROB.append(dict(predictor=pr, scorer=sc, n=n, n_changed=int(ch_state[pr].sum()), n_charge_changed=int(ch_charge[pr].sum()), n_fallback=int(fb[pr].sum()),
                        n_score_changed=int((np.abs(s_p - s_single) > TOL).sum()), Rp_single=Rp_s, Rp_pred=Rp_p, dRp=obs, lo=lo, hi=hi, p_boot=float(np.mean(bR <= 0)),
                        RMSE_single=RM_s, RMSE_pred=RM_p, dRMSE=obs_m, dRMSE_lo=mlo, dRMSE_hi=mhi,
                        p_null=pn["non_single"], p_null_incl_single=pn["incl_single"], p_null_changed_set=pn["changed_set"], n_matched=int(keep_mt.sum()), dRp_matched_obs=obs_mt, p_null_matched=pn["matched_dq"]))
        for dname, mask in (("state_differs", ch_state[pr]), ("charge_differs", ch_charge[pr]), ("score_changed", np.abs(s_p - s_single) > TOL)):
            m_ = int(mask.sum())
            if m_ < 3:
                AFFROWS.append(dict(predictor=pr, scorer=sc, definition=dname, n=m_)); continue
            ys, a_, b_ = y[mask], s_single[mask], s_p[mask]
            Ra, Ma = C.rp_rmse(ys, a_); Rb, Mb = C.rp_rmse(ys, b_)
            ix = C.boot_indices(m_, NB, ST["seed_boot"])
            sR, sM = C.paired_delta_boot(ys, a_, b_, ix)
            l1, h1 = C.pct(sR); l2, h2 = C.pct(sM)
            AFFROWS.append(dict(predictor=pr, scorer=sc, definition=dname, n=m_, Rp_single=Ra, Rp_pred=Rb, dRp=Rb - Ra, lo=l1, hi=h1, p_boot=float(np.mean(sR <= 0)),
                                RMSE_single=Ma, RMSE_pred=Mb, dRMSE=Mb - Ma, dRMSE_lo=l2, dRMSE_hi=h2))
    for p, m, s in zip(cl, sing, s_single):
        for pr in PRED:
            r = PS[(p, pr)]
            PCROWS.append(dict(pdbid=p, scorer=sc, predictor=pr, logKa=AFF[p], q_pred=r["q_pred"], single_q=m["q"], chosen_q=r["chosen_q"], status=r["status"], score_single=s,
                               score_pred=member(p, r["chosen_smiles"])["score"][sc], differs_from_single=r["differs_from_single"] == "True"))
    C.log(sc, "n", n, "Rp_single %.4f" % Rp_s, "original dRp dom %.4f rand %.4f" % (d_dom_old, d_rand_old), "null mean %.4f sd %.4f" % (d_ns.mean(), d_ns.std(ddof=1)), "%.0fs" % (time.time() - t0))

SC = [("scorer", "", "vina, smina, ad4"), ("null_type", "", "non_single / incl_single, changed_set (random only where the predictor changes the state), matched_dq"),
      ("predictor", "", "predictor whose observed dRp is compared with the null"), ("n_perturbed", "", "complexes in which a state is drawn / changed (matched_dq: complexes with a non-empty matched pool)"),
      ("n_draws", "", "number of random draws"), ("null_mean", "", "mean of dRp over the draws"), ("q025", "", "2.5th percentile of the null dRp"), ("q975", "", "97.5th percentile of the null dRp"),
      ("obs_dRp", "", "observed dRp(predictor state) - Rp(single); matched_dq: restricted to the complexes with a non-empty matched pool"), ("p_emp", "", "(#{null dRp >= obs} + 1) / (n_draws + 1)"),
      ("obs_within_null_95", "", "True when q025 <= obs_dRp <= q975"), ("frac_null_ge_obs", "", "#{null dRp >= obs} / n_draws")]
RC = [("predictor", "", "dimorphite_dom, openbabel, enum_popmax, ml_pka, consensus_ge3"), ("scorer", "", "vina, smina, ad4"), ("n", "", "n_common (complexes with finite scores for all seven original conditions)"),
      ("n_changed", "", "complexes whose chosen microstate differs from the default state"), ("n_charge_changed", "", "...with a different net charge"), ("n_fallback", "", "complexes keeping the default state because the predictor gave no usable state"),
      ("n_score_changed", "", "complexes whose score differs from the default state's by more than score_change_tol"), ("Rp_single", "", "Rp of the default state"), ("Rp_pred", "", "Rp of the predictor-defined state"),
      ("dRp", "", "Rp_pred - Rp_single"), ("lo", "", "paired bootstrap 2.5th percentile of dRp"), ("hi", "", "paired bootstrap 97.5th percentile of dRp"), ("p_boot", "", "fraction of bootstrap dRp <= 0 (one-sided, as in the original analysis)"),
      ("RMSE_single", "", "OLS RMSE of the default state"), ("RMSE_pred", "", "OLS RMSE of the predictor-defined state"), ("dRMSE", "", "RMSE_pred - RMSE_single"), ("dRMSE_lo", "", "bootstrap 2.5th percentile"), ("dRMSE_hi", "", "bootstrap 97.5th percentile"),
      ("p_null", "", "empirical p against the preset step-2 random null (non_single pool, every complex perturbed)"), ("p_null_incl_single", "", "same with the default state in the pool"),
      ("p_null_changed_set", "", "empirical p against the random null restricted to the complexes the predictor changes"), ("n_matched", "", "changed complexes with a non-empty matched-|dq| pool"),
      ("dRp_matched_obs", "", "observed dRp when only the n_matched complexes are changed"), ("p_null_matched", "", "empirical p against the matched-|dq| null")]
AC = [("predictor", "", "predictor"), ("scorer", "", "scorer"), ("definition", "", "state_differs (chosen microstate != default), charge_differs (net charge differs), score_changed (|score change| > tol; the original '94' for Vina and the Dimorphite dominant state)"),
      ("n", "", "complexes in the subset (within n_common)"), ("Rp_single", "", "Rp of the default state computed on the subset only"), ("Rp_pred", "", "Rp of the predictor state on the subset"), ("dRp", "", "difference"),
      ("lo", "", "paired bootstrap 2.5th percentile (complexes of the subset)"), ("hi", "", "paired bootstrap 97.5th percentile"), ("p_boot", "", "fraction of bootstrap dRp <= 0"), ("RMSE_single", "", "OLS RMSE on the subset"),
      ("RMSE_pred", "", "OLS RMSE on the subset"), ("dRMSE", "", "difference"), ("dRMSE_lo", "", "paired bootstrap 2.5th percentile of dRMSE"), ("dRMSE_hi", "", "paired bootstrap 97.5th percentile of dRMSE")]
PC = [("pdbid", "", "complex"), ("scorer", "", "scorer"), ("predictor", "", "predictor"), ("logKa", "", "experimental logKa (CoreSet.dat)"), ("q_pred", "e", "net charge predicted"), ("single_q", "e", "net charge of the default state"),
      ("chosen_q", "e", "net charge of the scored state"), ("status", "", "state-selection status (predictor_states.csv)"), ("score_single", "kcal/mol", "score of the default state"), ("score_pred", "kcal/mol", "score of the predictor-defined state"),
      ("differs_from_single", "", "chosen microstate is not the default")]
for name, cols_, rows, grain, desc in (("random_null_summary.csv", SC, SUMROWS, "scorer x null_type x predictor", "Random-microstate null distributions (mean, 2.5/97.5 percentiles) and empirical p of each predictor-defined dRp"),
                                       ("predictor_robustness.csv", RC, ROB, "predictor x scorer", "Gain of predictor-defined states over the default state, paired bootstrap CI, p against the nulls"),
                                       ("affected_subset.csv", AC, AFFROWS, "predictor x scorer x subset definition", "Gain restricted to the complexes in which the predictor changes the state / charge / score"),
                                       ("per_complex_scores_by_predictor.csv", PC, PCROWS, "complex x scorer x predictor (n_common)", "Per-complex default and predictor-defined scores used by the CASF-scoring statistics")):
    path = (C.RAW if name.startswith("per_complex") else C.RES) / name
    C.write_csv(path, cols_, rows)
    C.P.write_schema(path, cols_, grain, desc, extra=dict(provenance=BLOCK))
SUMM["failures"] = fail; SUMM["wall_seconds"] = round(time.time() - t0, 1)
json.dump(SUMM, open(C.RES / "random_null_summary_all.json", "w"), indent=2, ensure_ascii=False)
C.log("done; failures:", fail)
sys.exit(1 if fail else 0)
