import csv, hashlib, html, json, math, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import binomtest, fisher_exact

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
RM = Path(os.environ.get("BASE_ROOT", "."))
VF = Path(os.environ.get("DOCKING_ROOT", "."))
OUT_ROOT = Path(os.environ.get("TAUTOMER_OUT_ROOT", str(ROOT)))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts/utils")); sys.path.insert(0, str(HERE))
import provenance as P, tautomer_schemas as S, rdkit_version_comparison as RP

CFG = yaml.safe_load(open(ROOT / "configs/tautomer_ranking.yaml", encoding="utf-8"))
RES, RAW = OUT_ROOT / "results/tautomer_ranking", OUT_ROOT / "raw_outputs/tautomer_ranking"
RES.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
READ_LOG = set()


def E(s):
    return Path(s.replace("$BASE_ROOT", str(RM)).replace("$DOCKING_ROOT", str(VF)).replace("$ANALYSIS_ROOT", str(ROOT)))


PATHS = {k: E(v) for k, v in CFG["paths"].items()}
for k_, m_ in CFG["pinned_md5"].items():
    if P.file_md5(PATHS[k_]) != m_:
        sys.exit("pinned input changed: %s" % k_)
    READ_LOG.add(str(PATHS[k_]))
Z = NormalDist().inv_cdf(1 - CFG["stats"]["alpha"] / 2)
TOL = CFG["stats"]["energy_tie_tolerance_kcal"]
KPL = CFG["stats"]["kcal_per_logK"]
BAND_LO, BAND_HI = CFG["band"]["lo"], CFG["band"]["hi"]
FAILS = []


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def rd_tsv(path):
    READ_LOG.add(str(path))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def wilson(k, n):
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_paired(r, s, t, u):
    N = r + s + t + u; p1, p2 = (r + s) / N, (r + t) / N
    l1, u1 = wilson(r + s, N); l2, u2 = wilson(r + t, N)
    if 0 in (r + s, t + u, r + t, s + u): phi = 0.0
    else:
        A = (r + s) * (t + u) * (r + t) * (s + u); B = r * u - s * t
        C = B - N / 2 if B > N / 2 else (0.0 if B >= 0 else B); phi = C / math.sqrt(A)
    D = p1 - p2
    lo = D - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = D + math.sqrt((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2)
    return D, lo, hi, phi


def newcombe_unpaired(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); D = p1 - p2
    return D, D - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), D + math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)


def holm(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i]); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def band_relation(lo, hi):
    return "BELOW" if hi < BAND_LO else ("ABOVE" if lo > BAND_HI else "OVERLAP")


def point_position(rate):
    return "BELOW" if rate < BAND_LO else ("ABOVE" if rate > BAND_HI else "IN")


KNOWN = []
for ka in CFG["stats"]["newcombe_known_answers"]:
    _, lo_, hi_, phi_ = newcombe_paired(*ka["cells"])
    ok = abs(lo_ - ka["expected"][0]) <= ka["tol"] and abs(hi_ - ka["expected"][1]) <= ka["tol"]
    KNOWN.append(dict(cells=ka["cells"], expected=ka["expected"], got=[round(lo_, 6), round(hi_, 6)], phi=round(phi_, 4), tol=ka["tol"], ok=ok, source=ka["source"]))
    if not ok: FAILS.append("newcombe known answer %s" % (ka["cells"],))
g_ = CFG["stats"]["independent_known_answer"]["groups"]; _, lo_, hi_ = newcombe_unpaired(*g_)
ok = abs(lo_ - CFG["stats"]["independent_known_answer"]["expected"][0]) <= CFG["stats"]["independent_known_answer"]["tol"] and abs(hi_ - CFG["stats"]["independent_known_answer"]["expected"][1]) <= CFG["stats"]["independent_known_answer"]["tol"]
KNOWN.append(dict(cells=g_, expected=CFG["stats"]["independent_known_answer"]["expected"], got=[round(lo_, 6), round(hi_, 6)], phi=None, tol=CFG["stats"]["independent_known_answer"]["tol"], ok=ok, source="unpaired square-and-add of Wilson intervals = phi 0 case of the CAMIS example"))
if not ok: FAILS.append("unpaired known answer")
log("known-answer tests:", [k["ok"] for k in KNOWN])


pairs = rd_tsv(PATHS["pairs_tsv"]); PAIR = {r["row"]: r for r in pairs}
assert len(PAIR) == len(pairs) == 1677
sq = rd_tsv(PATHS["sqm_detail_tsv"]); xf = rd_tsv(PATHS["xtb_free_tsv"]); xo = rd_tsv(PATHS["xtb_opt_tsv"]); dft = rd_tsv(PATHS["dft_tsv"])
assert [r["row"] for r in sq] == [r["row"] for r in xf] == [r["row"] for r in xo] and len(sq) == 150 and len(dft) == 24
SQ, XF, XO, DFT = ({r["row"]: r for r in t} for t in (sq, xf, xo, dft))
READ_LOG.update(str(PATHS[k]) for k in ("tautobase_raw", "figbuild_xtb_json", "original_code_p2", "original_code_p3xtb", "original_code_p3dft"))

log("RDKit re-scoring (main environment)")
RS_ROWS, RDKIT_VERSION, RDKIT_DIGEST, RDKIT_SCORE_DIGEST = RP.rescore_all(PATHS["pairs_tsv"])
RS = {r["row"]: r for r in RS_ROWS}
CANON_BY_VERSION = {}
RDKIT_VERSIONS = {RDKIT_VERSION: dict(python="(main environment)", digest_sha256=RDKIT_DIGEST, score_digest_sha256=RDKIT_SCORE_DIGEST, n_canon_dom_differences_from_main=0, n_pairs=len(RS_ROWS), n_matching_stored=sum(1 for r in RS_ROWS if r["matches_stored"]), n_score_tie=sum(1 for r in RS_ROWS if r["score_dom"] == "tie"))}
for ver, py in CFG["rdkit_probe_pythons"].items():
    out = subprocess.run([str(E(py)), "-B", str(HERE / "rdkit_version_comparison.py"), str(PATHS["pairs_tsv"])], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if out.returncode: FAILS.append("rdkit probe %s: %s" % (ver, out.stderr[-200:])); continue
    d = json.loads(out.stdout.strip().splitlines()[-1]); CANON_BY_VERSION[d["rdkit_version"]] = d["canon_dom"]
    RDKIT_VERSIONS[d["rdkit_version"]] = dict(python=py, digest_sha256=d["digest_sha256"], score_digest_sha256=d["score_digest_sha256"], n_pairs=d["n_pairs"], n_matching_stored=d["n_matching_stored"], n_score_tie=d["n_score_tie"])
RDKIT_ALL_EQUAL = len({v["score_digest_sha256"] for v in RDKIT_VERSIONS.values()}) == 1
MAIN_CANON = {r["row"]: r["canon_dom"] for r in RS_ROWS}; FULL_ROWS_ = {r["row"] for r in pairs if r["measured"] == "True" and r["exp_dom"] in ("T1", "T2")}
CANON_DIFF = {}
for ver_, cd_ in CANON_BY_VERSION.items():
    dif = sorted(r for r in MAIN_CANON if cd_[r] != MAIN_CANON[r]); RDKIT_VERSIONS[ver_]["n_canon_dom_differences_from_main"] = len(dif)
    CANON_DIFF[ver_] = dict(rows=dif, n_in_full1101=sum(1 for r in dif if r in FULL_ROWS_), rows_in_the_150=[r for r in dif if r in {x["row"] for x in sq}])
mv = re.search(r"RDKit (\d{4}\.\d{2}\.\d+)", open(PATHS["manuscript_main"], encoding="utf-8").read()); MANUSCRIPT_RDKIT = mv.group(1) if mv else None
log("RDKit versions:", {k: (v["score_digest_sha256"][:10], v["n_canon_dom_differences_from_main"]) for k, v in RDKIT_VERSIONS.items()}, "score digests all equal:", RDKIT_ALL_EQUAL)
if not RDKIT_ALL_EQUAL: FAILS.append("RDKit ScoreTautomer differs between versions")
if RDKIT_VERSIONS[RDKIT_VERSION]["n_matching_stored"] != len(RS_ROWS): FAILS.append("RDKit re-scoring in the main environment differs from the stored columns")

def num(x):
    try: return float(x)
    except Exception: return None


FULL = [r for r in pairs if r["measured"] == "True" and r["exp_dom"] in ("T1", "T2")]
POOL = [r for r in FULL if num(r["logK_num"]) is not None and r["same_formula"] == "True" and int(r["heavy1"]) <= CFG["selection"]["sqm_heavy_max"]]


def stratified(elig, seed, max_pairs):
    rng = np.random.default_rng(seed); bins = defaultdict(list)
    for p in elig: bins[int(p["heavy1"])].append(p)
    keys = sorted(bins)
    for k in keys:
        b = bins[k]; bins[k] = [b[i] for i in rng.permutation(len(b))]
    cand = []
    while len(cand) < max_pairs and any(bins[k] for k in keys):
        for k in keys:
            if bins[k]:
                cand.append(bins[k].pop())
                if len(cand) >= max_pairs: break
    cand.sort(key=lambda p: int(p["heavy1"]))
    return [p["row"] for p in cand]


SEL150 = stratified(POOL, CFG["selection"]["sqm_seed"], CFG["selection"]["sqm_max_pairs"])
SEL150_OK = SEL150 == [r["row"] for r in sq]
water = [r for r in sq if r["solvent"] == CFG["selection"]["dft_solvent"]]
SEL24 = [r["row"] for r in sorted(water, key=lambda r: (int(r["heavy1"]), int(r["row"])))[:CFG["selection"]["dft_n"]]]
SEL24_OK = set(SEL24) == set(DFT)
log("pool", len(POOL), "full", len(FULL), "| 150 reproduced:", SEL150_OK, "| 24 reproduced:", SEL24_OK)
if not (SEL150_OK and SEL24_OK): FAILS.append("selection rules do not reproduce the stored subsets")

def status_E(dE, ok, exp_dom):
    if not ok: return "failed"
    if abs(dE) <= TOL: return "tie"
    return "correct" if ("T1" if dE > 0 else "T2") == exp_dom else "wrong"


def status_rdkit(row):
    d, e = RS[row]["score_dom"], PAIR[row]["exp_dom"]
    return "tie" if d == "tie" else ("correct" if d == e else "wrong")


CONSIST = Counter()
MAT = []
for r in sq:
    row = r["row"]; p, f, o = PAIR[row], XF[row], XO[row]; d = DFT.get(row)
    CONSIST["exp_dom differs (sqm vs pairs)"] += r["exp_dom"] != p["exp_dom"]
    CONSIST["exp_dom differs (xtb free/opt vs pairs)"] += (f["exp_dom"] != p["exp_dom"]) + (o["exp_dom"] != p["exp_dom"])
    CONSIST["t1/t2 differ (sqm vs pairs)"] += (r["t1"] != p["t1"]) + (r["t2"] != p["t2"])
    CONSIST["PM6 energy differs between sqm and xtb files"] += abs(float(f["dHf_pm6"]) - float(r["dHf_pm6"])) > 1e-9
    CONSIST["dG_exp != -1.3642*logK"] += abs(float(r["dG_exp"]) + KPL * float(r["logK_num"])) > 1e-6
    CONSIST["stored xtb_dom != recomputed"] += (f["xtb_dom"] != ("T1" if float(f["dE_xtb"]) > 0 else "T2")) + (o["xtb_dom"] != ("T1" if float(o["dE_xtb"]) > 0 else "T2"))
    CONSIST["stored score_dom (sqm file) != recomputed RDKit"] += r["score_dom"] != RS[row]["score_dom"]
    CONSIST["exp_src not logK"] += p["exp_src"] != "logK"
    if d: CONSIST["stored dft_dom != recomputed"] += d["dft_dom"] != ("T1" if float(d["dE_dft"]) > 0 else "T2")
    MAT.append(dict(row=row, heavy1=p["heavy1"], solvent=p["solvent"], exp_dom=p["exp_dom"], exp_src=p["exp_src"], logK_num=p["logK_num"],
                    dG_exp_kcal=format(-KPL * float(p["logK_num"]), ".12g"), in_dft24=str(row in DFT),
                    rdkit_score_t1=RS[row]["score_t1"], rdkit_score_t2=RS[row]["score_t2"], rdkit_status=status_rdkit(row),
                    pm6_dE_kcal=r["dHf_pm6"], pm6_status=status_E(float(r["dHf_pm6"]), r["sqm_ok"] == "True", p["exp_dom"]),
                    gfn2_free_dE_kcal=f["dE_xtb"], gfn2_free_status=status_E(float(f["dE_xtb"]), f["xtb_ok"] == "True", p["exp_dom"]),
                    gfn2_opt_dE_kcal=o["dE_xtb"], gfn2_opt_status=status_E(float(o["dE_xtb"]), o["xtb_ok"] == "True", p["exp_dom"]),
                    dft_dE_kcal=d["dE_dft"] if d else "", dft_status=status_E(float(d["dE_dft"]), d["dft_ok"] == "True", p["exp_dom"]) if d else "",
                    t1_smiles=p["t1"], t2_smiles=p["t2"]))
CONSIST = {k: int(v) for k, v in CONSIST.items()}
log("consistency counters (all must be 0):", CONSIST)
if any(CONSIST.values()): FAILS.append("input consistency counters non-zero: %s" % CONSIST)
MROW = {m["row"]: m for m in MAT}


def count(rows, key, val): return sum(1 for m in rows if m[key] == val)


REP = CFG["reported_values"]; REPORTED = []


def leg(name, expected, got):
    ok = expected == got; REPORTED.append(dict(name=name, expected=expected, got=got, ok=ok))
    if not ok: FAILS.append("reported: %s expected %s got %s" % (name, expected, got))


leg("RDKit on the 150: correct/wrong/tie", REP["rdkit_150"], dict(correct=count(MAT, "rdkit_status", "correct"), wrong=count(MAT, "rdkit_status", "wrong"), tie=count(MAT, "rdkit_status", "tie")))
leg("PM6 wrong on the 150", REP["pm6_150_wrong"], count(MAT, "pm6_status", "wrong"))
leg("GFN2-xTB --ohess wrong on the 150", REP["gfn2_free_150_wrong"], count(MAT, "gfn2_free_status", "wrong"))
leg("GFN2-xTB --opt wrong on the 150", REP["gfn2_opt_150_wrong"], count(MAT, "gfn2_opt_status", "wrong"))
M24 = [m for m in MAT if m["in_dft24"] == "True"]
leg("DFT wrong on the 24", REP["dft24_wrong"], count(M24, "dft_status", "wrong"))
leg("PM6 wrong on the 24", REP["pm6_24_wrong"], count(M24, "pm6_status", "wrong"))
leg("GFN2-xTB --opt wrong on the 24 (the original DFT trend table used the --opt energies)", REP["gfn2_opt_24_wrong"], count(M24, "gfn2_opt_status", "wrong"))
leg("GFN2-xTB --ohess wrong on the 24 (not in the original table; the manuscript's 150-pair xTB variant)", REP["gfn2_free_24_wrong"], count(M24, "gfn2_free_status", "wrong"))
for nm, key, dE in (("pm6_gas", "pm6_dE_kcal", None), ("gfn2_free", "gfn2_free_dE_kcal", None), ("gfn2_opt", "gfn2_opt_dE_kcal", None)):
    rmse = math.sqrt(sum((float(m[key]) - float(m["dG_exp_kcal"])) ** 2 for m in MAT) / len(MAT)); exp_ = REP["rmse_kcal"][nm]
    dec = len(str(exp_).split(".")[1]); leg("RMSE of %s on the 150 (rounded to the reported precision)" % nm, exp_, round(rmse, dec))

METHODS = [("RDKit_score", "rdkit_status"), ("PM6_gas", "pm6_status"), ("GFN2xtb_ALPB_free", "gfn2_free_status"), ("GFN2xtb_ALPB_opt", "gfn2_opt_status")]
PRIMARY = [("RDKit_score", "rdkit_status"), ("PM6_gas", "pm6_status"), ("GFN2xtb_ALPB_free", "gfn2_free_status")]
COMMON = [m for m in MAT if all(m[k] in ("correct", "wrong") for _, k in PRIMARY)]
RATE_ROWS = []; RATE = {}


def add_rate(method, conv, k, n, n_tie, n_fail, note="wilson"):
    lo, hi = wilson(k, n); rate = 100 * k / n
    row = dict(method=method, convention=conv, k=("%.1f" % k if conv == "D" else str(int(k))), n=n, rate_pct="%.4f" % rate, wilson_lo_pct="%.4f" % (100 * lo), wilson_hi_pct="%.4f" % (100 * hi),
               n_tie=n_tie, n_failed=n_fail, band_lo_pct="%g" % BAND_LO, band_hi_pct="%g" % BAND_HI, band_relation=band_relation(100 * lo, 100 * hi), point_position=point_position(rate), ci_note=note)
    RATE_ROWS.append(row); RATE[(method, conv)] = dict(k=k, n=n, rate=rate, lo=100 * lo, hi=100 * hi, rel=row["band_relation"], pos=row["point_position"])


for conv in ("A", "B", "C", "D"):
    for name, key in METHODS:
        n_tie, n_fail = count(MAT, key, "tie"), count(MAT, key, "failed")
        if conv == "A": add_rate(name, "A", len(MAT) - count(MAT, key, "correct"), len(MAT), n_tie, n_fail)
        elif conv == "B": add_rate(name, "B", count(COMMON, key, "wrong"), len(COMMON), n_tie, n_fail)
        elif conv == "C":
            keep = [m for m in MAT if m[key] in ("correct", "wrong")]; add_rate(name, "C", count(keep, key, "wrong"), len(keep), n_tie, n_fail)
        else: add_rate(name, "D", count(MAT, key, "wrong") + n_fail + 0.5 * n_tie, len(MAT), n_tie, n_fail, "wilson on the effective count")
log("n common subset (convention B):", len(COMMON))

BOOT_RNG = np.random.default_rng(CFG["stats"]["seed_bootstrap"]); NB = CFG["stats"]["n_boot"]; CH = CFG["stats"]["boot_chunk"]
MC_ROWS = []; MC = {}


def boot_diff(r, s, t, u):
    N = r + s + t + u; probs = np.array([r, s, t, u]) / N; out = []
    for _ in range(NB // CH):
        c = BOOT_RNG.multinomial(N, probs, size=CH); out.append((c[:, 1] - c[:, 2]) / N)
    d = np.concatenate(out)
    return float(np.percentile(d, CFG["stats"]["boot_percentiles"][0])), float(np.percentile(d, CFG["stats"]["boot_percentiles"][1]))


for conv, rows in (("A", MAT), ("B", COMMON)):
    for family, trio in (("primary", PRIMARY), ("sensitivity_opt", [PRIMARY[0], PRIMARY[1], METHODS[3]])):
        batch = []
        for i in range(3):
            for j in range(i + 1, 3):
                (n1, k1), (n2, k2) = trio[i], trio[j]
                w1 = [m[k1] != "correct" if conv == "A" else m[k1] == "wrong" for m in rows]; w2 = [m[k2] != "correct" if conv == "A" else m[k2] == "wrong" for m in rows]
                r = sum(a and b for a, b in zip(w1, w2)); s = sum(a and not b for a, b in zip(w1, w2)); t = sum(b and not a for a, b in zip(w1, w2)); u = len(rows) - r - s - t
                D, lo, hi, phi = newcombe_paired(r, s, t, u); blo, bhi = boot_diff(r, s, t, u)
                batch.append(dict(convention=conv, family=family, method_1=n1, method_2=n2, n=len(rows), both_wrong=r, b_only_1_wrong=s, c_only_2_wrong=t, both_right=u, p_exact=mcnemar_exact(s, t),
                                  rate_1_pct="%.4f" % (100 * (r + s) / len(rows)), rate_2_pct="%.4f" % (100 * (r + t) / len(rows)), diff_pct_points="%.4f" % (100 * D),
                                  newcombe_lo="%.4f" % (100 * lo), newcombe_hi="%.4f" % (100 * hi), boot_lo="%.4f" % (100 * blo), boot_hi="%.4f" % (100 * bhi), n_boot=NB,
                                  note=("b + c < %d: only the exact test is meaningful" % CFG["stats"]["min_discordant_for_asymptotic"]) if s + t < CFG["stats"]["min_discordant_for_asymptotic"] else ""))
        assert len(batch) == CFG["stats"]["holm_family_size"]
        for row, ph in zip(batch, holm([b["p_exact"] for b in batch])):
            row["p_holm"] = "%.6g" % ph; row["p_exact"] = "%.6g" % row["p_exact"]
            MC_ROWS.append(row); MC[(row["convention"], row["family"], row["method_1"], row["method_2"])] = row

HB = [tuple(b) for b in CFG["representativeness"]["heavy_bins"]]


def hbin(h): return next("%d-%d" % b for b in HB if b[0] <= h <= b[1])


def rd_state(r, conv):
    sd, cd, e = RS[r["row"]]["score_dom"], RS[r["row"]]["canon_dom"], r["exp_dom"]
    if conv == "score_strict": return 0 if sd == e else 1
    if conv == "score_lenient": return None if sd == "tie" else (0 if sd == e else 1)
    if conv == "canon_strict": return 0 if cd == e else 1
    return None if cd == "other" else (0 if cd == e else 1)


SUB150 = set(SEL150); POOLROWS = {r["row"] for r in POOL}
POPS = [("full1101", FULL), ("pool332", POOL), ("subset150", [r for r in POOL if r["row"] in SUB150]), ("pool_remainder182", [r for r in POOL if r["row"] not in SUB150])]
REPR_ROWS = []; REPRK = {}
CONVS = ("score_strict", "score_lenient", "canon_strict", "canon_lenient")


def add_repr(pop, stype, stratum, rows):
    for cv in CONVS:
        st = [rd_state(r, cv) for r in rows]; st = [x for x in st if x is not None]
        if not st: continue
        k, n = sum(st), len(st); lo, hi = wilson(k, n)
        REPR_ROWS.append(dict(population=pop, stratum_type=stype, stratum=stratum, convention=cv, k=k, n=n, rate_pct="%.4f" % (100 * k / n), wilson_lo_pct="%.4f" % (100 * lo), wilson_hi_pct="%.4f" % (100 * hi)))
        REPRK[(pop, stype, stratum, cv)] = (k, n)


for pop, rows in POPS:
    add_repr(pop, "all", "all", rows)
    for b in HB:
        sel = [r for r in rows if b[0] <= int(r["heavy1"]) <= b[1]]
        if sel: add_repr(pop, "heavy_bin", "%d-%d" % b, sel)
for src in ("logK", "Preferred", "%T1", "logK_ineq", "%T1_ineq"):
    add_repr("full1101", "exp_src", src, [r for r in FULL if r["exp_src"] == src])

FS = REP["rdkit_full1101"]
leg("full set: n with a clear dominant tautomer", FS["n_clear"], len(FULL))
leg("full set: evidence-class composition", FS["exp_source_composition"], dict(Counter(r["exp_src"] for r in FULL)))
for cv, key in (("score_strict", "score_strict_pct"), ("score_lenient", "score_lenient_pct"), ("canon_strict", "canon_strict_pct"), ("canon_lenient", "canon_lenient_pct")):
    k, n = REPRK[("full1101", "all", "all", cv)]; leg("full set RDKit %s rate (rounded to 2 decimals)" % cv, FS[key], round(100 * k / n, 2))
leg("logK-class RDKit score_lenient rate = original summary by_exp_source (24.81)", 24.81, round(100 * REPRK[("full1101", "exp_src", "logK", "score_lenient")][0] / REPRK[("full1101", "exp_src", "logK", "score_lenient")][1], 2))

TEST_ROWS = []


def fisher_row(tid, desc, g1, g2, cv, k1, n1, k2, n2, note):
    D, lo, hi = newcombe_unpaired(k1, n1, k2, n2); p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    TEST_ROWS.append(dict(test_id=tid, test_type="fisher_exact", description=desc, group_1=g1, k1=k1, n1=n1, rate1_pct="%.4f" % (100 * k1 / n1), group_2=g2, k2=k2, n2=n2, rate2_pct="%.4f" % (100 * k2 / n2),
                          effect_type="pct_points", effect="%.4f" % (100 * D), effect_lo="%.4f" % (100 * lo), effect_hi="%.4f" % (100 * hi), p_value=p, p_holm="", note=note))


def grp(pop, cv): return REPRK[(pop, "all", "all", cv)]


def minus(a, b): return (a[0] - b[0], a[1] - b[1])


full_ = {cv: grp("full1101", cv) for cv in ("score_lenient", "score_strict")}
tid = 1
for cv in ("score_lenient", "score_strict"):
    s150, rem, pool, full = grp("subset150", cv), grp("pool_remainder182", cv), grp("pool332", cv), full_[cv]
    fisher_row("R%d" % tid, "RDKit %s: the 150 against the rest of their own 332-pair pool" % cv, "subset150", "pool_remainder182", cv, *s150, *rem, "is the selection of the 150 representative of its pool"); tid += 1
    fisher_row("R%d" % tid, "RDKit %s: the numeric-logK pool of 332 against the other 769 pairs of the full set" % cv, "pool332", "full_minus_pool769", cv, *pool, *minus(full, pool), "evidence class and size cap differ"); tid += 1
    fisher_row("R%d" % tid, "RDKit %s: the 150 against the other 951 pairs of the full set" % cv, "subset150", "full_minus_subset951", cv, *s150, *minus(full, s150), "the headline gap between 26%/40% and 39%/50%"); tid += 1
pv = [float(r["p_value"]) for r in TEST_ROWS]
for r, ph in zip(TEST_ROWS, holm(pv)): r["p_holm"] = "%.6g" % ph; r["p_value"] = "%.6g" % r["p_value"]
LDF = pd.DataFrame([dict(wrong=rd_state(r, "score_lenient"), sub=int(r["row"] in SUB150), src=("ineq" if r["exp_src"].endswith("_ineq") else r["exp_src"]), heavy=int(r["heavy1"])) for r in FULL if rd_state(r, "score_lenient") is not None])
n_dec, k_sub, n_sub = len(LDF), int(LDF[LDF["sub"] == 1]["wrong"].sum()), int((LDF["sub"] == 1).sum())
for lid, formula, desc, note in (("L1", "wrong ~ sub", "crude odds ratio of being misranked, subset150 against the other decisions of the full set", "crude"),
                                 ("L2", "wrong ~ sub + C(src, Treatment('Preferred')) + heavy", "same, adjusted for evidence class and heavy-atom count", "adjusted for evidence class (logK / Preferred / %T1 / inequality) and heavy-atom count")):
    fit = smf.logit(formula, LDF).fit(disp=0, maxiter=200); ci = fit.conf_int().loc["sub"]
    conv_ok = bool(fit.mle_retvals.get("converged", False))
    if not conv_ok: FAILS.append("logistic regression %s did not converge" % lid)
    TEST_ROWS.append(dict(test_id=lid, test_type="logit", description=desc + " (RDKit score_lenient)", group_1="subset150", k1=k_sub, n1=n_sub, rate1_pct="%.4f" % (100 * k_sub / n_sub), group_2="full_minus_subset", k2=int(LDF["wrong"].sum()) - k_sub,
                          n2=n_dec - n_sub, rate2_pct="%.4f" % (100 * (int(LDF["wrong"].sum()) - k_sub) / (n_dec - n_sub)), effect_type="odds_ratio", effect="%.4f" % math.exp(fit.params["sub"]), effect_lo="%.4f" % math.exp(ci[0]),
                          effect_hi="%.4f" % math.exp(ci[1]), p_value="%.6g" % fit.pvalues["sub"], p_holm="", note=note + ("; converged" if conv_ok else "; NOT converged")))

D24 = [m for m in MAT if m["in_dft24"] == "True"]
D24_REST = [m for m in MAT if m["in_dft24"] != "True"]
DFT_PAIR_ROWS = [dict(row=m["row"], heavy1=m["heavy1"], solvent=m["solvent"], exp_dom=m["exp_dom"], logK_num=m["logK_num"], dG_exp_kcal=m["dG_exp_kcal"], rdkit_status=m["rdkit_status"], pm6_status=m["pm6_status"],
                       gfn2_free_status=m["gfn2_free_status"], gfn2_opt_status=m["gfn2_opt_status"], dft_status=m["dft_status"], pm6_dE_kcal=m["pm6_dE_kcal"], gfn2_free_dE_kcal=m["gfn2_free_dE_kcal"], dft_dE_kcal=m["dft_dE_kcal"]) for m in D24]
DCMP = []
M5 = [("RDKit_score", "rdkit_status", "A"), ("RDKit_score", "rdkit_status", "ties_dropped"), ("PM6_gas", "pm6_status", "A"), ("GFN2xtb_ALPB_free", "gfn2_free_status", "A"), ("GFN2xtb_ALPB_opt", "gfn2_opt_status", "A")]


def is_wrong(m, key, conv): return m[key] != "correct" if conv == "A" else m[key] == "wrong"


def sel_conv(rows, key, conv): return [m for m in rows if conv == "A" or m[key] != "tie"]


for name, key, conv in M5 + [("DFT_SMD", "dft_status", "A")]:
    rows = sel_conv(D24, key, conv); k = sum(is_wrong(m, key, conv) for m in rows); lo, hi = wilson(k, len(rows))
    DCMP.append(dict(analysis="rate", convention=conv, method=name, group_1="dft24", k1=k, n1=len(rows), rate1_pct="%.4f" % (100 * k / len(rows)), wilson1_lo_pct="%.4f" % (100 * lo), wilson1_hi_pct="%.4f" % (100 * hi)))
for conv in ("A", "ties_dropped"):
    batch = []
    for name, key, cv in [x for x in M5 if x[2] == conv]:
        rows = sel_conv(D24, key, cv); w1 = [is_wrong(m, key, cv) for m in rows]; w2 = [m["dft_status"] != "correct" for m in rows]
        r = sum(a and b for a, b in zip(w1, w2)); s = sum(a and not b for a, b in zip(w1, w2)); t = sum(b and not a for a, b in zip(w1, w2)); u = len(rows) - r - s - t
        D, lo, hi, _ = newcombe_paired(r, s, t, u)
        batch.append(dict(analysis="paired_vs_dft", convention=conv, method=name, group_1="method", k1=r + s, n1=len(rows), rate1_pct="%.4f" % (100 * (r + s) / len(rows)), group_2="DFT_SMD", k2=r + t, n2=len(rows), rate2_pct="%.4f" % (100 * (r + t) / len(rows)),
                          b=s, c=t, diff_pct_points="%.4f" % (100 * D), diff_lo="%.4f" % (100 * lo), diff_hi="%.4f" % (100 * hi), p_value=mcnemar_exact(s, t)))
    for row, ph in zip(batch, holm([b["p_value"] for b in batch])): row["p_holm"] = "%.6g" % ph; row["p_value"] = "%.6g" % row["p_value"]; DCMP.append(row)
for conv in ("A", "ties_dropped"):
    batch = []
    for name, key, cv in [x for x in M5 if x[2] == conv]:
        a_, b_ = sel_conv(D24, key, cv), sel_conv(D24_REST, key, cv); k1 = sum(is_wrong(m, key, cv) for m in a_); k2 = sum(is_wrong(m, key, cv) for m in b_)
        D, lo, hi = newcombe_unpaired(k1, len(a_), k2, len(b_)); lo1, hi1 = wilson(k1, len(a_))
        batch.append(dict(analysis="dft24_vs_rest", convention=conv, method=name, group_1="dft24", k1=k1, n1=len(a_), rate1_pct="%.4f" % (100 * k1 / len(a_)), wilson1_lo_pct="%.4f" % (100 * lo1), wilson1_hi_pct="%.4f" % (100 * hi1),
                          group_2="other_pairs_of_the_150", k2=k2, n2=len(b_), rate2_pct="%.4f" % (100 * k2 / len(b_)), diff_pct_points="%.4f" % (100 * D), diff_lo="%.4f" % (100 * lo), diff_hi="%.4f" % (100 * hi),
                          p_value=float(fisher_exact([[k1, len(a_) - k1], [k2, len(b_) - k2]])[1])))
    for row, ph in zip(batch, holm([b["p_value"] for b in batch])): row["p_holm"] = "%.6g" % ph; row["p_value"] = "%.6g" % row["p_value"]; DCMP.append(row)
D24_HEAVY = sorted(int(m["heavy1"]) for m in D24)

def text_of(path):
    READ_LOG.add(str(path)); t = open(path, encoding="utf-8", errors="replace").read()
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S); t = html.unescape(re.sub(r"<[^>]+>", " ", t)); return re.sub(r"\s+", " ", t)


def excerpt(t, pat, w=260, n=3):
    return [t[max(0, m.start() - w): m.end() + w] for m in list(re.finditer(pat, t))[:n]]


perry = text_of(PATHS["lit_perry_html"]); wieder = text_of(PATHS["lit_wieder_html"])
perry_rec = json.load(open(PATHS["lit_perry_json"], encoding="utf-8"))["resultList"]["result"][0]; READ_LOG.add(str(PATHS["lit_perry_json"]))
ref_start = perry.find("(1). ", 5000); perry_body = perry[:ref_start] if ref_start > 5000 else perry
TERMS = {"Tautobase": r"Tautobase", "implicit solvent": r"implicit[- ]solv", "continuum solvation": r"continuum solv", "SMD": r"\bSMD\b", "COSMO": r"COSMO", "PCM": r"\bPCM\b", "ALPB": r"ALPB", "tautomer ratio": r"tautomer ratio", "logK": r"log ?K",
         "experimental ratio": r"experimental (tautomer )?ratio", "polymorph": r"polymorph", "periodic": r"periodic", "DLPNO-CCSD(T1)": r"DLPNO-CCSD", "gas-phase": r"gas[- ]phase", "NMR": r"NMR"}
term_counts = {k: len(re.findall(p, perry_body, flags=re.I)) for k, p in TERMS.items()}
hits = [i.start() for i in re.finditer(r"20\s?[-–]\s?30\s?%", perry_body)]
mw = re.search(r"For (\d+) tautomer pairs \((\d+)% of the dataset\) the method was not able to correctly determine the dominant tautomer", wieder)
mw2 = re.search(r"468 tautomer pairs \((\d+) if the basis set 6-31G\(d\) was used\)", wieder)
alt_k, alt_n = (int(mw.group(1)), int(mw2.group(1))) if mw and mw2 else (None, None)
if (alt_k, alt_n) != (CFG["alt_reference"]["wrong_pairs"], CFG["alt_reference"]["n_pairs"]): FAILS.append("Wieder 2021 numbers not found as configured")
alt_lo, alt_hi = wilson(CFG["alt_reference"]["wrong_pairs"], CFG["alt_reference"]["n_pairs"])
main_txt = open(PATHS["manuscript_main"], encoding="utf-8").read(); si_txt = open(PATHS["supplementary"], encoding="utf-8").read()
BAND_SOURCE = dict(
    status="TRACED_TO_A_SOURCE_BUT_THE_SOURCE_DOES_NOT_MATCH_THE_MANUSCRIPT_DESCRIPTION",
    conclusion=("The only literature item carrying a '20-30%' figure for DFT errors that could be traced is Perry et al., J. Am. Chem. Soc. 2025, 147, 26865 (the original code tags it 'JACS2025_DFT_wrong_dominant_frac'). "
                "It reports that periodic DFT models give the wrong qualitative STABILITY ORDERING OF TAUTOMERIC CRYSTAL POLYMORPHS in ~20-30% of the surveyed cases, against a DLPNO-CCSD(T1)-corrected reference. "
                "It is not a solution-phase benchmark, uses no implicit solvent, no Tautobase and no experimental tautomer ratios, so it is not a band of 'solvated density-functional treatments' of dominant-tautomer misranking. "
                "The manuscript gives no citation for the band. The band must therefore be treated as UNVERIFIED in its current wording (reported to the user; the text is not rewritten here)."),
    manuscript_text=dict(main_sentence=excerpt(main_txt, r"every method sits above the 20–30% band", 380, 1), si_sentence=excerpt(si_txt, r"remain in the 20–30% band", 300, 1),
                         citation_keys_in_main_paragraph=sorted(set(re.findall(r"\[@[^\]]+\]", next(p for p in main_txt.split("\n") if "every method sits above" in p)))), cites_perry_beran_or_wieder=any(w in main_txt + si_txt for w in ("Beran", "Perry", "Wieder", "Chodera"))),
    original_code_tag=dict(file="code/analysis/tautomer_benchmark.py and p3_tautomer_solvated_xtb.py", key="literature_ref.JACS2025_DFT_wrong_dominant_frac", value="0.20-0.30"),
    traced_source=dict(authors=perry_rec.get("authorString"), title=perry_rec.get("title"), journal="J. Am. Chem. Soc.", year=perry_rec.get("pubYear"), volume=147, issue=30, pages=perry_rec.get("pageInfo"), doi=perry_rec.get("doi"), pmcid=perry_rec.get("pmcid"),
                       abstract_sentence=excerpt(perry_rec.get("abstractText", ""), r"20-30%", 220, 1), full_text_hits_of_20_30pct=[perry_body[max(0, h - 260): h + 200] for h in hits],
                       survey_size="the abstract says 'Surveying 18 additional examples' (tautomeric crystal polymorph systems); the per-case table is in the paper's SI, which was not retrieved"),
    equivalence_check_term_counts_in_paper_body=term_counts, reference_list_starts_at_char=ref_start,
    why_not_equivalent=["subject: crystal polymorph relative energies, not dominant tautomer in solution", "method: periodic DFT (GGA / hybrid) with XDM dispersion, judged against DLPNO-CCSD(T1)-corrected gas-phase intramolecular energies, not against experimental tautomer ratios",
                        "no implicit-solvent treatment and no Tautobase appear in the paper body (counts above)", "the quantity is a rate of wrong qualitative stability ordering of polymorphs, not a rate of misranked dominant tautomers"],
    like_for_like_literature_value=dict(source_key=CFG["alt_reference"]["source_key"], citation="Wieder M, Fass J, Chodera JD. Chem. Sci. 2021, 12, 11364 (doi 10.1039/d1sc01185e)", method="B3LYP/aug-cc-pVTZ // B3LYP/6-31G(d) + SMD(water) transfer free energy, RRHO gas-phase free energies, Tautobase aqueous pairs with numeric logK",
                                        wrong_dominant_pairs=alt_k, n_pairs=alt_n, rate_pct=round(100 * alt_k / alt_n, 2), wilson_pct=[round(100 * alt_lo, 2), round(100 * alt_hi, 2)], quote=excerpt(wieder, r"For \d+ tautomer pairs \(\d+% of the dataset\)", 20, 1),
                                        caveats="a different pair set (460 of the Tautobase aqueous pairs, not our 150); position-only comparison; DFT here still has 13% wrong dominant tautomers, below the band the manuscript quotes; not used for the band relation in misrank_rates.csv"),
    band_relation_rule="BELOW if the Wilson upper end is under the band, ABOVE if the Wilson lower end is over it, else OVERLAP; point_position compares the point estimate (band edges inclusive: IN)",
    sources_saved_in=dict(perry_html=str(PATHS["lit_perry_html"]), perry_record=str(PATHS["lit_perry_json"]), wieder_html=str(PATHS["lit_wieder_html"])),
)

def fmt(v): return "" if v is None else str(v)


def write_csv(path, cols, rows):
    names = [c[0] for c in cols]
    for r in rows:
        if set(r) - set(names): sys.exit("%s: row has keys that are not schema columns: %s" % (path.name, sorted(set(r) - set(names))))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(names)
        for r in rows: w.writerow([fmt(r.get(n)) for n in names])


SCRIPT = os.path.abspath(__file__)
INPUTS = sorted(READ_LOG)
PROV_EXTRA = dict(role="tautomer-ranking tautomer 150-subset recheck", numpy=np.__version__, pandas=pd.__version__, null_fields_explained=dict(n_perm="no permutation test is used: McNemar and Fisher tests are exact, the paired CI is Newcombe's closed form plus a multinomial bootstrap"),
                  bootstrap_unit="tautomer pair (paired multinomial resampling of the four cell counts)", rdkit_probe_digests={k: v["digest_sha256"] for k, v in RDKIT_VERSIONS.items()}, out_root=str(OUT_ROOT))
BLOCK = P.provenance_block(SCRIPT, ROOT / "configs/tautomer_ranking.yaml", INPUTS, seed=CFG["stats"]["seed_bootstrap"], n_boot=NB, n_perm=None, extra=PROV_EXTRA, tools_extra=("statsmodels", "rdkit"))
JOBS = [
    (RES / "tautomer_per_pair_matrix.csv", S.MATRIX, MAT, "Per-pair statuses of the 150 pairs behind the manuscript's PM6 / GFN2-xTB / RDKit rates, re-derived from the stored energies and a fresh RDKit scoring"),
    (RES / "misrank_rates.csv", S.RATES, RATE_ROWS, "Misranking rate of each method on the 150 pairs under four counting conventions, with Wilson 95% intervals and their relation to the literature band"),
    (RES / "mcnemar.csv", S.MCNEMAR, MC_ROWS, "Paired comparisons of the three main methods (exact McNemar, Holm over three, Newcombe and bootstrap intervals of the rate difference)"),
    (RES / "subset_representativeness.csv", S.REPR, REPR_ROWS, "RDKit misranking rates in the full 1,101-pair set, the 332-pair numeric-logK pool, the 150-pair subset and the rest of the pool, stratified by evidence class and heavy-atom bin"),
    (RES / "subset_representativeness_tests.csv", S.REPR_TESTS, TEST_ROWS, "Two-group comparisons (Fisher exact, Holm over six) and logistic regressions asking whether the 150-pair subset differs from the rest of the data"),
    (RES / "dft24_same_pairs.csv", S.DFT_PAIRS, DFT_PAIR_ROWS, "The 24 pairs that were also computed with DFT: per-pair statuses of every method"),
    (RES / "dft24_comparisons.csv", S.DFT_CMP, DCMP, "Method rates on the 24 DFT pairs, paired comparisons with DFT, and the 24 pairs against the other 126 of the 150"),
]
for path, cols, rows, desc in JOBS:
    write_csv(path, cols, rows); P.write_schema(path, cols, S.ROW_GRAIN[path.name], desc, extra=dict(provenance=BLOCK))
RESCORE_ROWS = [dict(row=r["row"], score_t1=r["score_t1"], score_t2=r["score_t2"], score_dom=r["score_dom"], canon_dom=r["canon_dom"], matches_stored=str(r["matches_stored"])) for r in RS_ROWS]
write_csv(RAW / "rdkit_rescore_all.csv", S.RESCORE, RESCORE_ROWS)
P.write_schema(RAW / "rdkit_rescore_all.csv", S.RESCORE, S.ROW_GRAIN["rdkit_rescore_all.csv"], "RDKit scores and canonical tautomer of every Tautobase pair of the original analysis, recomputed in the analysis environment", extra=dict(provenance=BLOCK))
(RES / "band_source.json").write_text(json.dumps(dict(BAND_SOURCE, provenance=BLOCK), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

def rate_fact(m, c): r = RATE[(m, c)]; return dict(k=r["k"], n=r["n"], rate_pct=round(r["rate"], 2), wilson_pct=[round(r["lo"], 2), round(r["hi"], 2)], band_relation=r["rel"], point_position=r["pos"])


ANSWERS = dict(
    q1_is_26pct_above_the_20_30_band={c: {m: rate_fact(m, c) for m, _ in METHODS} for c in ("A", "B", "C", "D")},
    q2_is_rdkit_no_worse_than_semiempirical={c: [dict(comparison="%s vs %s" % (r["method_1"], r["method_2"]), b=r["b_only_1_wrong"], c=r["c_only_2_wrong"], diff_pct_points=float(r["diff_pct_points"]), newcombe=[float(r["newcombe_lo"]), float(r["newcombe_hi"])], p_exact=float(r["p_exact"]), p_holm=float(r["p_holm"]))
                                                for r in MC_ROWS if r["convention"] == c and r["family"] == "primary"] for c in ("A", "B")},
)
figj = json.load(open(PATHS["figbuild_xtb_json"], encoding="utf-8")); READ_LOG.add(str(PATHS["figbuild_xtb_json"]))
RM_ = {nm: math.sqrt(sum((float(m[k]) - float(m["dG_exp_kcal"])) ** 2 for m in MAT) / len(MAT)) for nm, k in (("pm6_gas", "pm6_dE_kcal"), ("gfn2_free", "gfn2_free_dE_kcal"), ("gfn2_opt", "gfn2_opt_dE_kcal"))}
XTB_EVIDENCE = dict(figure_data_method=figj["provenance"]["method"], figure_data_rmse=figj["xtb_all150"]["dG_RMSE_kcal"], figure_data_misselect_pct=figj["xtb_all150"]["misselect_pct"], recomputed_rmse={k: round(v, 4) for k, v in RM_.items()},
                    manuscript_states="31% / 30.7% misranked, free-energy RMSE 5.4 kcal/mol", conclusion="the manuscript's xTB numbers are the --ohess (Gibbs free energy) variant: RMSE 5.44 -> 5.4, whereas --opt would read 5.5; both variants misrank 46/150 (30.67%), with 43 wrong pairs in common")
wf, wo = {m["row"] for m in MAT if m["gfn2_free_status"] == "wrong"}, {m["row"] for m in MAT if m["gfn2_opt_status"] == "wrong"}
XTB_EVIDENCE["wrong_pairs_in_common"] = len(wf & wo)
SUMMARY = dict(
    n_pairs_150=len(MAT), n_common_subset_B=len(COMMON), n_pool=len(POOL), n_full=len(FULL), n_dft24=len(D24), dft24_heavy_atoms=dict(Counter(D24_HEAVY)), n_failures_or_ties={k: dict(tie=count(MAT, v, "tie"), failed=count(MAT, v, "failed")) for k, v in METHODS},
    conventions=CFG["conventions"], dft24_selection_rule=CFG["selection"]["dft_order"], selection_reproduced=dict(sqm150=SEL150_OK, dft24=SEL24_OK, numpy=np.__version__), input_consistency_counters=CONSIST, reported_reproduction=REPORTED, known_answer_tests=KNOWN,
    rdkit=dict(main_version=RDKIT_VERSION, manuscript_states=MANUSCRIPT_RDKIT, versions=RDKIT_VERSIONS, canon_dom_differences=CANON_DIFF, all_digests_equal=RDKIT_ALL_EQUAL, n_pairs_rescored=len(RS_ROWS), n_matching_stored=RDKIT_VERSIONS[RDKIT_VERSION]["n_matching_stored"]),
    xtb_variant_evidence=XTB_EVIDENCE, band=dict(lo=BAND_LO, hi=BAND_HI, status=BAND_SOURCE["status"], like_for_like_literature=BAND_SOURCE["like_for_like_literature_value"]), answers=ANSWERS, failures=FAILS,
    n_rows=dict(matrix=len(MAT), rates=len(RATE_ROWS), mcnemar=len(MC_ROWS), representativeness=len(REPR_ROWS), tests=len(TEST_ROWS), dft_pairs=len(DFT_PAIR_ROWS), dft_comparisons=len(DCMP), rescore=len(RS_ROWS)), provenance=BLOCK)
(RES / "tautomer_summary.json").write_text(json.dumps(SUMMARY, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def r2(x): return "%.2f" % float(x)


def md_table(head, rows): return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


kt = ["# tautomer-ranking key tables (generated by tautomer.py; do not edit)\n", "## Misranking rates on the 150 pairs (k/n, rate, Wilson 95% CI, relation to the 20-30% band)\n",
      md_table(["convention", "method", "k/n", "rate %", "Wilson 95% CI %", "band relation", "point"], [[r["convention"], r["method"], "%s/%s" % (r["k"], r["n"]), r2(r["rate_pct"]), "%s - %s" % (r2(r["wilson_lo_pct"]), r2(r["wilson_hi_pct"])), r["band_relation"], r["point_position"]] for r in RATE_ROWS]),
      "\n\n## Paired comparisons (exact McNemar, Holm over 3, Newcombe 95% CI of the rate difference in percentage points)\n",
      md_table(["conv", "family", "method 1 vs 2", "n", "b / c", "diff pp", "Newcombe CI", "p_exact", "p_Holm"], [[r["convention"], r["family"], "%s vs %s" % (r["method_1"], r["method_2"]), r["n"], "%s / %s" % (r["b_only_1_wrong"], r["c_only_2_wrong"]), r2(r["diff_pct_points"]), "%s - %s" % (r2(r["newcombe_lo"]), r2(r["newcombe_hi"])), r["p_exact"], r["p_holm"]] for r in MC_ROWS]),
      "\n\n## Subset representativeness (RDKit)\n", md_table(["population", "stratum", "convention", "k/n", "rate %", "Wilson 95% CI %"], [[r["population"], "%s=%s" % (r["stratum_type"], r["stratum"]), r["convention"], "%s/%s" % (r["k"], r["n"]), r2(r["rate_pct"]), "%s - %s" % (r2(r["wilson_lo_pct"]), r2(r["wilson_hi_pct"]))] for r in REPR_ROWS if r["convention"] in ("score_lenient", "score_strict")]),
      "\n\n## Representativeness tests\n", md_table(["id", "groups", "rate1 / rate2 %", "effect", "95% CI", "p", "p_Holm"], [[r["test_id"], "%s vs %s" % (r["group_1"], r["group_2"]), "%s / %s" % (r2(r["rate1_pct"]), r2(r["rate2_pct"])), "%s %s" % (r["effect_type"], r2(r["effect"])), "%s - %s" % (r2(r["effect_lo"]), r2(r["effect_hi"])), r["p_value"], r["p_holm"]] for r in TEST_ROWS]),
      "\n\n## DFT-24 comparisons\n", md_table(["analysis", "conv", "method", "k1/n1", "rate1 %", "k2/n2", "diff pp (CI)", "b / c", "p", "p_Holm"], [[r["analysis"], r["convention"], r["method"], "%s/%s" % (r["k1"], r["n1"]), r2(r["rate1_pct"]), ("%s/%s" % (r["k2"], r["n2"])) if r.get("k2", "") != "" else "", ("%s (%s - %s)" % (r2(r["diff_pct_points"]), r2(r["diff_lo"]), r2(r["diff_hi"]))) if r.get("diff_pct_points", "") != "" else "", ("%s / %s" % (r["b"], r["c"])) if r.get("b", "") != "" else "", r.get("p_value", ""), r.get("p_holm", "")] for r in [dict({"k2": "", "diff_pct_points": "", "b": "", "p_value": "", "p_holm": ""}, **x) for x in DCMP]]), "\n"]
log("wrote", len(JOBS), "result CSVs; failures:", FAILS)
sys.exit(1 if FAILS else 0)
