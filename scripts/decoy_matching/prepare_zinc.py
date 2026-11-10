import os
import argparse, collections, gzip, hashlib, json, multiprocessing as mp, random, re, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts" / "utils"))
import provenance as P

SEED = 20261016
BASE = "https://files.docking.org/2D"
SCRIPT = ROOT / "scripts/decoy_matching/prepare_zinc.py"
ZDIR = ROOT / "inputs/external/zinc20"
RAW = ZDIR / "raw"
LDIR = ROOT / "logs/downloads"
LISTINGS = LDIR / "listings"
RES = ROOT / "results/decoy_matching"
INDEX = ZDIR / "zinc20_2d_file_index.tsv"
PLAN = ZDIR / "zinc_file_plan.tsv"
DLFILES = ZDIR / "zinc_downloaded_files.tsv"
MD5LIST = ZDIR / "zinc20_raw_md5.txt"
POOL = ZDIR / "zinc_pool.tsv"
MAPGZ = ZDIR / "zinc_raw_to_pool_map.tsv.gz"
BUILDLOG = RES / "zinc_pool_build_log.json"
COVERAGE = RES / "zinc_pool_coverage.tsv"
MANIFEST = ROOT / "inputs/manifests/DATA_MANIFEST.tsv"
MANIFEST_BAK = Path("/tmp/DATA_MANIFEST.bak_zinc")

ALL_LETTERS = "ABCDEFGHIJK"
MW_LETTERS, LP_LETTERS, REACT_OK, PURCH_OK = "CDEFGHIJ", "BCDEFGHIJ", "ABCE", "AB"
MW_EDGES = dict(zip(ALL_LETTERS, [(0, 200), (200, 250), (250, 300), (300, 325), (325, 350), (350, 375), (375, 400), (400, 425), (425, 450), (450, 500), (500, 1e9)]))
LP_EDGES = dict(zip(ALL_LETTERS, [(-1e9, -1), (-1, 0), (0, 1), (1, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 4.5), (4.5, 5), (5, 1e9)]))
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
HEAVY_MIN, HEAVY_MAX = 15, 40
HEADER_PREFIX = "smiles\tzinc_id\tinchikey\tmwt\tlogp\treactive\tpurchasable\ttranche_name"

ROW_RX = re.compile(r'<a href="([A-K]{4})\.(smi|txt)">[^<]*</a>\s*</td><td align="right">([0-9-]+ [0-9:]+)\s*</td>'
                    r'<td align="right">\s*([0-9.]+)([KMGT]?)\s*</td>')
UNIT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
_lock = threading.Lock()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sh(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def eligible(name):
    return name[0] in MW_LETTERS and name[1] in LP_LETTERS and name[2] in REACT_OK and name[3] in PURCH_OK


def append_line(path, line):
    with _lock:
        with open(path, "a") as f:
            f.write(line + "\n")


def stage_list(a):
    LISTINGS.mkdir(parents=True, exist_ok=True)
    log = LDIR / "listing_fetch_log.tsv"
    dirs = [x + y for x in ALL_LETTERS for y in ALL_LETTERS]

    def one(d):
        out = LISTINGS / (d + ".html")
        if out.is_file() and b"Index of" in out.read_bytes()[:4000]:
            return d, "present"
        url = "%s/%s/" % (BASE, d)
        for k in range(1, 6):
            ts = now()
            r = sh(["curl", "-sS", "-L", "--connect-timeout", "30", "--max-time", "120", "-o", str(out), "-w", "%{http_code}", url])
            code = r.stdout.strip()[-3:] or "000"
            append_line(log, "%s\t%s\t%s\t%s\tlist_stage_try%d\t%s" % (ts, code, url, d, k, r.stderr.strip()[:150].replace("\n", " ")))
            if code == "200" and b"Index of" in out.read_bytes()[:4000]:
                return d, "fetched"
            time.sleep(4 * k)
        return d, "FAILED"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, dirs))
    print("listing status:", dict(collections.Counter(s for _, s in res)))
    if any(s == "FAILED" for _, s in res):
        sys.exit("listing fetch failed for: %s" % [d for d, s in res if s == "FAILED"])
    rows, n_href = [], 0
    for d in dirs:
        h = (LISTINGS / (d + ".html")).read_text(errors="replace")
        n_href += len(re.findall(r'href="[A-K]{4}\.(?:smi|txt)"', h))
        for m in ROW_RX.finditer(h):
            name, ext, mtime, sz, u = m.groups()
            assert name[:2] == d
            rows.append((name, ext, mtime, sz + u, int(float(sz) * UNIT[u])))
    assert len(rows) == n_href, "listing parser matched %d of %d file hrefs" % (len(rows), n_href)
    rows.sort()
    with open(INDEX, "w") as f:
        f.write("tranche\tcell\tmw_letter\tlogp_letter\treact_letter\tpurch_letter\text\tmtime\tsize_text\tsize_bytes_approx\n")
        for name, ext, mtime, st, sb in rows:
            f.write("\t".join([name, name[:2], name[0], name[1], name[2], name[3], ext, mtime, st, str(sb)]) + "\n")
    print("index rows %d (smi %d, txt %d) -> %s" % (len(rows), sum(r[1] == "smi" for r in rows), sum(r[1] == "txt" for r in rows), INDEX))


def read_index():
    out = []
    with open(INDEX) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            out.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    return out


def stage_plan(a):
    idx = [r for r in read_index() if r["ext"] == "txt" and eligible(r["tranche"])]
    cells = collections.defaultdict(list)
    for r in idx:
        cells[r["cell"]].append(r)
    rng = random.Random(SEED)
    rows = []
    for cell in sorted(cells):
        fl = sorted(cells[cell], key=lambda r: r["tranche"])
        rng.shuffle(fl)
        for k, r in enumerate(fl, 1):
            rows.append((cell, r["tranche"], k, r["size_bytes_approx"], "%s/%s/%s.txt" % (BASE, cell, r["tranche"])))
    with open(PLAN, "w") as f:
        f.write("cell\ttranche\tround\tsize_bytes_approx\turl\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    cnt = collections.Counter(r[2] for r in rows)
    print("plan: %d eligible .txt files in %d cells (expected %d x %d = %d); files per round: %s; seed %d"
          % (len(rows), len(cells), len(MW_LETTERS), len(LP_LETTERS), len(MW_LETTERS) * len(LP_LETTERS), dict(sorted(cnt.items())), SEED))


def read_plan():
    out = []
    with open(PLAN) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            r["round"] = int(r["round"])
            out.append(r)
    return out


def validate_raw(path):
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with open(p, "rb") as f:
        head = f.readline().decode("utf-8", "replace").rstrip("\n")
        if not head.startswith(HEADER_PREFIX):
            return None
        n = 0
        last = b""
        for chunk in iter(lambda: f.read(1 << 22), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        return None
    return n if n >= 1 else None


def parse_head(text):
    blocks = [b for b in re.split(r"\r?\n\r?\n", text.strip()) if b.strip()]
    if not blocks:
        return None, None
    last = blocks[-1].splitlines()
    m = re.match(r"HTTP/\S+\s+(\d+)", last[0])
    code = int(m.group(1)) if m else None
    cl = None
    for ln in last[1:]:
        if ln.lower().startswith("content-length:"):
            cl = int(ln.split(":", 1)[1].strip())
    return cl, code


def download_one(t, log):
    cell = t[:2]
    url = "%s/%s/%s.txt" % (BASE, cell, t)
    dst, part = RAW / (t + ".txt"), RAW / (t + ".txt.part")
    if dst.is_file() and validate_raw(dst) is not None:
        return t, "present"
    if dst.is_file():
        dst.unlink()
    ts0, t0 = now(), time.time()
    exp, hcode = None, None
    for k in range(4):
        r = sh(["curl", "-sSIL", "--connect-timeout", "30", "--max-time", "120", url])
        exp, hcode = parse_head(r.stdout)
        if hcode == 200 and exp:
            break
        time.sleep(3 * (k + 1))
    attempts, http, err = 0, "000", ""
    if hcode == 200 and exp:
        while attempts < 6:
            attempts += 1
            if part.exists() and part.stat().st_size > exp:
                part.unlink()
            if part.exists() and part.stat().st_size == exp:
                break
            r = sh(["curl", "-sS", "-L", "-C", "-", "--connect-timeout", "30", "--max-time", "900", "-o", str(part), "-w", "%{http_code}", url])
            http, err = (r.stdout.strip()[-3:] or "000"), r.stderr.strip()[:150].replace("\n", " ")
            if part.exists() and part.stat().st_size == exp:
                break
            time.sleep(3 * attempts)
    size = part.stat().st_size if part.exists() else 0
    status, md5 = "FAILED", ""
    if exp and size == exp:
        part.replace(dst)
        if validate_raw(dst) is not None:
            status, md5 = "OK", P.file_md5(dst)
        else:
            status = "FAILED_VALIDATION"
    append_line(log, "\t".join([ts0, now(), t, url, str(hcode), http, str(exp), str(size), str(attempts), "%.1f" % (time.time() - t0), md5, status, err]))
    return t, status


def stage_download(a):
    RAW.mkdir(parents=True, exist_ok=True)
    LDIR.mkdir(parents=True, exist_ok=True)
    log = LDIR / "download_log.tsv"
    if not log.exists():
        log.write_text("ts_start\tts_end\ttranche\turl\thead_http\tget_http\texpected_bytes\tbytes\tattempts\tsecs\tmd5\tstatus\tcurl_err\n")
    plan = read_plan()
    rounds = sorted({r["round"] for r in plan})
    byround = {k: [r["tranche"] for r in plan if r["round"] == k] for k in rounds}

    def cumulative():
        tot = 0
        for r in plan:
            n = validate_raw(RAW / (r["tranche"] + ".txt"))
            tot += n or 0
        return tot

    for k in rounds[: a.max_round]:
        cum = cumulative()
        if cum >= a.target_raw_rows:
            print("round %d not needed: cumulative raw rows %d >= target %d" % (k, cum, a.target_raw_rows))
            break
        print("[%s] round %d: %d files (cumulative raw rows before: %d, target %d)" % (now(), k, len(byround[k]), cum, a.target_raw_rows), flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(lambda t: download_one(t, log), byround[k]))
        c = collections.Counter(s for _, s in res)
        print("[%s] round %d status: %s" % (now(), k, dict(c)), flush=True)
        if any(s.startswith("FAILED") for _, s in res):
            sys.exit("round %d has failed files: %s" % (k, [t for t, s in res if s.startswith("FAILED")]))
    cum = cumulative()
    rows, nbytes = [], 0
    for r in plan:
        p = RAW / (r["tranche"] + ".txt")
        n = validate_raw(p)
        if n is None:
            continue
        b = p.stat().st_size
        nbytes += b
        rows.append((r["tranche"], r["cell"], r["round"], b, n, P.file_md5(p)))
    rows.sort()
    with open(DLFILES, "w") as f:
        f.write("tranche\tcell\tround\tbytes\tn_rows\tmd5\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(MD5LIST, "w") as f:
        for r in rows:
            f.write("%s  raw/%s.txt\n" % (r[5], r[0]))
    print("downloaded files %d, %.3f GB, raw rows %d -> %s, %s" % (len(rows), nbytes / 1e9, cum, DLFILES, MD5LIST))


def work(smis):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append(("parse_fail", "", 0, 0.0, "", 0, 0))
            continue
        frags = Chem.GetMolFrags(m, asMols=True)
        nfrag = len(frags)
        if nfrag > 1:
            frag = max(frags, key=lambda x: (x.GetNumHeavyAtoms(), Descriptors.MolWt(x), Chem.MolToSmiles(x)))
        else:
            frag = frags[0] if frags else m
        heavy = frag.GetNumHeavyAtoms()
        if heavy == 0:
            out.append(("no_atoms", "", 0, 0.0, "", nfrag, 0))
            continue
        mw = round(Descriptors.MolWt(frag), 3)
        if not (HEAVY_MIN <= heavy <= HEAVY_MAX):
            out.append(("heavy_out", "", heavy, mw, "", nfrag, 0))
            continue
        if not {a.GetSymbol() for a in frag.GetAtoms()} <= ALLOWED_ELEMENTS:
            out.append(("element_out", "", heavy, mw, "", nfrag, 0))
            continue
        try:
            key = Chem.MolToInchiKey(frag)
        except Exception:
            key = ""
        if not key or len(key) < 14:
            out.append(("inchikey_fail", "", heavy, mw, "", nfrag, 0))
            continue
        charged = int(any(a.GetFormalCharge() != 0 for a in frag.GetAtoms()))
        out.append(("ok", Chem.MolToSmiles(frag), heavy, mw, key[:14], nfrag, charged))
    return out


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def stage_build(a):
    import numpy as np
    from rdkit import Chem, rdBase
    from rdkit.Chem import inchi
    assert inchi.INCHI_AVAILABLE, "RDKit build has no InChI support"
    t_start = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    dl = []
    with open(DLFILES) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            dl.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    dl.sort(key=lambda r: r["tranche"])
    recs = []
    n_lines, n_malformed = 0, 0
    malformed = []
    for r in dl:
        p = RAW / (r["tranche"] + ".txt")
        with open(p, encoding="utf-8", errors="replace") as f:
            f.readline()
            for ln, line in enumerate(f, 2):
                n_lines += 1
                fl = line.rstrip("\n").split("\t")
                if len(fl) < 8 or not fl[0].strip() or not fl[1].strip():
                    n_malformed += 1
                    malformed.append((r["tranche"], ln, line.rstrip("\n")))
                    continue
                recs.append((r["tranche"], ln, fl[1].strip(), fl[0].strip(), fl[2], fnum(fl[3]), fnum(fl[4]), fl[7]))
    assert n_lines == sum(int(r["n_rows"]) for r in dl), "row count from files != downloaded-file table"
    assert n_lines == len(recs) + n_malformed
    bin_check = {"mw": {}, "logp": {}}
    for axis, pos, edges, vi in (("mw", 0, MW_EDGES, 5), ("logp", 1, LP_EDGES, 6)):
        for L in sorted({r[0][pos] for r in recs}):
            v = np.array([r[vi] for r in recs if r[0][pos] == L], dtype=float)
            v = v[~np.isnan(v)]
            lo, hi = edges[L]
            bin_check[axis][L] = dict(n=int(v.size), nominal_lo_excl=lo, nominal_hi_incl=hi, observed_min=float(v.min()), q001=float(np.quantile(v, 0.001)),
                                      median=float(np.median(v)), q999=float(np.quantile(v, 0.999)), observed_max=float(v.max()),
                                      frac_outside_nominal=float(((v <= lo - 1e-9) | (v > hi + 1e-9)).mean()))
    seen_id, status, dup_of = {}, [None] * len(recs), [""] * len(recs)
    keep_idx = []
    for i, r in enumerate(recs):
        if r[2] in seen_id:
            status[i] = "drop_dup_zinc_id"
            dup_of[i] = recs[seen_id[r[2]]][2] + "@" + recs[seen_id[r[2]]][0]
        else:
            seen_id[r[2]] = i
            keep_idx.append(i)
    n_dup_id = len(recs) - len(keep_idx)
    chunk = 4000
    smis = [recs[i][3] for i in keep_idx]
    batches = [smis[k:k + chunk] for k in range(0, len(smis), chunk)]
    ctx = mp.get_context("fork")
    res = []
    t0 = time.time()
    with ctx.Pool(a.procs) as pool:
        for k, out in enumerate(pool.imap(work, batches, chunksize=1)):
            res.extend(out)
            if k % 25 == 0:
                print("[%s] RDKit batches %d/%d (%.0fs)" % (now(), k + 1, len(batches), time.time() - t0), flush=True)
    assert len(res) == len(keep_idx)
    cleaned = {}
    cnt = collections.Counter()
    n_multi, n_charged = 0, 0
    for i, o in zip(keep_idx, res):
        st, smi, heavy, mw, skel, nfrag, charged = o
        cnt[st] += 1
        if nfrag > 1:
            n_multi += 1
        if st == "ok":
            cleaned[i] = (smi, heavy, mw, skel, charged)
            n_charged += charged
        else:
            status[i] = "drop_" + st
    first_by_skel, n_dup_skel = {}, 0
    pool_rows = []
    for i in keep_idx:
        if i in cleaned:
            smi, heavy, mw, skel, charged = cleaned[i]
            if skel in first_by_skel:
                status[i] = "drop_dup_inchikey_skeleton"
                dup_of[i] = recs[first_by_skel[skel]][2]
                n_dup_skel += 1
            else:
                first_by_skel[skel] = i
                status[i] = "kept"
                pool_rows.append(i)
    n0 = n_lines
    steps = []

    def add(name, desc, n_in, n_drop, breakdown=None):
        steps.append(dict(step=name, description=desc, n_in=n_in, n_dropped=n_drop, n_out=n_in - n_drop, **({"breakdown": breakdown} if breakdown else {})))

    add("00_read_records", "data lines of all downloaded tranche .txt files; dropped = lines with <8 tab fields or empty SMILES/ZINC id", n0, n_malformed)
    add("01_dedup_zinc_id", "drop repeated ZINC id (first occurrence in tranche/line order kept)", steps[-1]["n_out"], n_dup_id)
    add("02_rdkit_parse", "Chem.MolFromSmiles with default sanitisation", steps[-1]["n_out"], cnt["parse_fail"])
    add("03_largest_fragment", "keep largest fragment (max heavy atoms, then MW, then SMILES); salts/counter-ions removed, none neutralised; dropped = no heavy atom left", steps[-1]["n_out"], cnt["no_atoms"])
    add("04_heavy_atoms_%d_%d" % (HEAVY_MIN, HEAVY_MAX), "heavy atoms of the kept fragment within [%d,%d]" % (HEAVY_MIN, HEAVY_MAX), steps[-1]["n_out"], cnt["heavy_out"])
    add("05_elements", "kept fragment contains only %s" % sorted(ALLOWED_ELEMENTS), steps[-1]["n_out"], cnt["element_out"])
    add("06_inchikey", "standard InChIKey generated by RDKit/InChI", steps[-1]["n_out"], cnt["inchikey_fail"])
    add("07_dedup_inchikey_skeleton", "drop repeated first InChIKey block (14 chars); first occurrence kept", steps[-1]["n_out"], n_dup_skel)
    for k, s in enumerate(steps):
        assert s["n_in"] == s["n_dropped"] + s["n_out"], s
        if k:
            assert s["n_in"] == steps[k - 1]["n_out"], (s, steps[k - 1])
    assert steps[-1]["n_out"] == len(pool_rows) == len(first_by_skel)
    assert collections.Counter(s for s in status if s is not None).get("kept", 0) == len(pool_rows)
    rng = random.Random(SEED)
    order = list(pool_rows)
    rng.shuffle(order)
    cols = ["zinc_id", "smiles", "inchikey_skeleton", "heavy_atoms", "mw", "tranche", "source_file"]
    mism_tr = 0
    with open(POOL, "w") as f:
        f.write("\t".join(cols) + "\n")
        for i in order:
            tr, ln, zid, raw, zkey, zmw, zlp, ztr = recs[i]
            smi, heavy, mw, skel, charged = cleaned[i]
            if ztr != tr:
                mism_tr += 1
            row = [zid, smi, skel, str(heavy), "%.3f" % mw, tr, "raw/%s.txt" % tr]
            assert not any(("\t" in x or "\n" in x) for x in row)
            f.write("\t".join(row) + "\n")
    with gzip.open(MAPGZ, "wt", compresslevel=6) as g:
        g.write("zinc_id\tsource_file\tline_no\traw_smiles\tzinc_inchikey\tzinc_mwt\tzinc_logp\tstatus\tinchikey_skeleton\tduplicate_of\n")
        ki = {i: o for i, o in zip(keep_idx, res)}
        for i, r in enumerate(recs):
            tr, ln, zid, raw, zkey, zmw, zlp, ztr = r
            skel = cleaned[i][3] if i in cleaned else ""
            g.write("\t".join([zid, "raw/%s.txt" % tr, str(ln), raw, zkey, "%g" % zmw, "%g" % zlp, status[i], skel, dup_of[i]]) + "\n")
        for tr, ln, line in malformed:
            g.write("\t".join(["", "raw/%s.txt" % tr, str(ln), line.replace("\t", " | "), "", "", "", "drop_malformed", "", ""]) + "\n")
    n_map = len(recs) + len(malformed)
    assert n_map == n_lines
    P.write_schema(POOL, [
        ("zinc_id", "id", "ZINC20 numeric id exactly as distributed in the tranche .txt (ZINC20 display form is 'ZINC' + id zero-padded to 12 digits)"),
        ("smiles", "SMILES", "RDKit canonical isomeric SMILES of the largest fragment (salts removed, charge state as distributed, not neutralised)"),
        ("inchikey_skeleton", "InChIKey block 1", "first 14 characters of the standard InChIKey of the kept fragment; unique in this table"),
        ("heavy_atoms", "count", "heavy-atom count of the kept fragment (15-40 by construction)"),
        ("mw", "Da", "RDKit average molecular weight of the kept fragment (Descriptors.MolWt)"),
        ("tranche", "4-letter code", "ZINC20 tranche = MW bin, logP bin, reactivity, purchasability (e.g. EEAB); first two letters = MW x logP cell"),
        ("source_file", "path", "downloaded file under inputs/external/zinc20/ that the record came from"),
    ], row_grain="one drug-like in-stock ZINC20 molecule (deduplicated on the InChIKey skeleton)",
        description="D5 unmatched decoy pool: random draw of ZINC20 2D tranche files (seed %d), in stock (purchasability A,B), reactivity A,B,C,E, MW 250-500 / logP -1..5 tranches; rows shuffled with the seed" % SEED,
        extra=dict(provenance=P.provenance_block(SCRIPT, None, [INDEX, PLAN, DLFILES, MD5LIST], SEED, tools_extra=("rdkit",)),
                   raw_to_pool_map=str(MAPGZ), build_log=str(BUILDLOG)))
    pool_cell = collections.Counter(recs[i][0][:2] for i in pool_rows)
    raw_cell = collections.Counter(r[0][:2] for r in recs)
    files_cell = collections.Counter(r["cell"] for r in dl)
    plan_cell = collections.Counter(r["cell"] for r in read_plan())
    with open(COVERAGE, "w") as f:
        f.write("cell\tmw_letter\tlogp_letter\tmw_range_da\tlogp_range\tn_files_planned\tn_files_downloaded\tn_raw_rows\tn_pool\n")
        for m in MW_LETTERS:
            for l in LP_LETTERS:
                c = m + l
                f.write("\t".join([c, m, l, "(%g,%g]" % MW_EDGES[m], "(%g,%g]" % LP_EDGES[l], str(plan_cell[c]), str(files_cell[c]), str(raw_cell[c]), str(pool_cell[c])]) + "\n")
    grid = ["pool molecules per cell (rows logP letter B..J, cols MW letter C..J)", "      " + " ".join("%8s" % m for m in MW_LETTERS)]
    for l in LP_LETTERS:
        grid.append("%4s  " % l + " ".join("%8d" % pool_cell[m + l] for m in MW_LETTERS))
    print("\n".join(grid))
    mw_arr = np.array([cleaned[i][2] for i in pool_rows])
    heavy_arr = np.array([cleaned[i][1] for i in pool_rows])
    agree, comparable = 0, 0
    for i in pool_rows:
        if "." not in recs[i][3] and len(recs[i][4]) >= 14:
            comparable += 1
            agree += int(recs[i][4][:14] == cleaned[i][3])
    extra = dict(
        n_multi_fragment_inputs=n_multi, n_pool_with_formal_charge=int(sum(cleaned[i][4] for i in pool_rows)),
        pool_mw=dict(min=float(mw_arr.min()), q01=float(np.quantile(mw_arr, .01)), median=float(np.median(mw_arr)), q99=float(np.quantile(mw_arr, .99)), max=float(mw_arr.max()),
                     n_outside_250_500=int(((mw_arr < 250) | (mw_arr > 500)).sum())),
        pool_heavy_atoms=dict(min=int(heavy_arr.min()), median=float(np.median(heavy_arr)), max=int(heavy_arr.max())),
        zinc_vs_rdkit_inchikey_skeleton_agreement=dict(n_comparable_single_fragment=comparable, n_agree=agree, frac=(agree / comparable if comparable else None)),
        n_record_tranche_name_differs_from_file=mism_tr,
        n_pool_cells_nonempty=sum(1 for c in pool_cell.values() if c > 0), n_cells_expected=len(MW_LETTERS) * len(LP_LETTERS),
        min_cell_pool_n=min(pool_cell[m + l] for m in MW_LETTERS for l in LP_LETTERS), max_cell_pool_n=max(pool_cell.values()))
    log_rows = [json.loads(json.dumps(dict(zip(("ts_start", "ts_end", "tranche", "url", "head_http", "get_http", "expected_bytes", "bytes", "attempts", "secs", "md5", "status", "err"),
                                                  ln.rstrip("\n").split("\t") + [""] * 13)))) for ln in open(LDIR / "download_log.tsv").readlines()[1:]]
    ok_rows = [r for r in log_rows if r["status"] == "OK"]
    dsum = dict(n_files=len(dl), total_bytes=sum(int(r["bytes"]) for r in dl), total_gb=round(sum(int(r["bytes"]) for r in dl) / 1e9, 4),
                n_rounds_downloaded=len({r["round"] for r in dl}), rounds=sorted({int(r["round"]) for r in dl}),
                first_download_start=min((r["ts_start"] for r in ok_rows), default=None), last_download_end=max((r["ts_end"] for r in ok_rows), default=None),
                n_download_log_entries=len(log_rows), n_download_log_failed=sum(1 for r in log_rows if r["status"] != "OK"),
                n_files_needing_more_than_1_attempt=sum(1 for r in ok_rows if int(r["attempts"] or 0) > 1))
    outputs = {}
    for p in (POOL, POOL.with_name("zinc_pool.schema.json"), MAPGZ, COVERAGE, MD5LIST, INDEX, PLAN, DLFILES):
        outputs[str(p)] = dict(bytes=p.stat().st_size, md5=P.file_md5(p), sha256=sha256(p))
    outputs[str(POOL)]["n_rows"] = len(pool_rows)
    outputs[str(MAPGZ)]["n_rows"] = n_map
    blk = P.provenance_block(SCRIPT, None, [INDEX, PLAN, DLFILES, MD5LIST], SEED, extra=dict(stage="build", procs=a.procs, rdkit=rdBase.rdkitVersion), tools_extra=("rdkit",))
    log = dict(provenance=blk,
               source=dict(base=BASE, listing_dir="logs/downloads/listings", notes="ZINC20 2D tranche static export; purchasability letters A,B = in stock; zinc20.docking.org tranche browser is behind a captcha and was not used"),
               selection=dict(mw_letters=MW_LETTERS, logp_letters=LP_LETTERS, reactivity_letters=REACT_OK, purchasability_letters=PURCH_OK, seed=SEED,
                              design="per MW x logP cell the eligible files are shuffled with the seed; complete rounds (one file per cell) were downloaded until raw rows >= target"),
               download=dsum, steps=steps, step_sums_verified=True,
               n_pool=len(pool_rows), min_required=300000, meets_min_required=len(pool_rows) >= 300000,
               extra_counts=extra, bin_check=bin_check, outputs=outputs, runtime_s=round(time.time() - t_start, 1))
    BUILDLOG.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    print("steps:")
    for s in steps:
        print("  %-28s n_in=%8d dropped=%7d n_out=%8d" % (s["step"], s["n_in"], s["n_dropped"], s["n_out"]))
    print("pool %d (>=300000: %s); log %s" % (len(pool_rows), len(pool_rows) >= 300000, BUILDLOG))
    print(json.dumps(extra, indent=1))


def stage_manifest(a):
    log = json.loads(BUILDLOG.read_text())
    lines = MANIFEST.read_text().splitlines()
    hdr = lines[0].split("\t")
    exist = {ln.split("\t")[0] for ln in lines[1:]}
    if not MANIFEST_BAK.exists():
        shutil.copy2(MANIFEST, MANIFEST_BAK)
    d = log["download"]
    o = log["outputs"]
    listing_bytes = sum(p.stat().st_size for p in LISTINGS.glob("*.html"))
    rows = [
        ["D5_zinc20_2d_directory_listings", BASE + "/<MW><logP>/ (121 directory index pages)", "curl on server; 8 parallel first pass (3 transient HTTP 000, retried OK), see logs/downloads/listing_fetch_log.tsv",
         d["first_download_start"][:10] if d["first_download_start"] else time.strftime("%F"), "logs/downloads/listings", str(listing_bytes), "md5 per file not listed (index pages); parsed index md5=" + o[str(INDEX)]["md5"],
         "121", str(len(list(LISTINGS.glob("*.html")))), "OK", "parsed into inputs/external/zinc20/zinc20_2d_file_index.tsv (4275 smi + 4275 txt entries)"],
        ["D5_zinc20_2d_tranche_txt", BASE + "/<MW><logP>/<tranche>.txt", "curl -C - (resumable), 4 connections, scripts/decoy_matching/prepare_zinc.py download; log logs/downloads/download_log.tsv",
         d["first_download_start"] or "", "inputs/external/zinc20/raw", str(d["total_bytes"]), "per-file md5 in inputs/external/zinc20/zinc20_raw_md5.txt (md5 of that list=%s)" % o[str(MD5LIST)]["md5"],
         "%d (seeded rounds %s of %d planned eligible files)" % (d["n_files"], d["rounds"], len(read_plan())), str(d["n_files"]), "OK",
         "ZINC20 2D tranches, in stock (A,B), reactivity A,B,C,E, MW C-J, logP B-J; start %s end %s; seed %d" % (d["first_download_start"], d["last_download_end"], SEED)],
        ["D5_zinc_pool.tsv", "derived from D5_zinc20_2d_tranche_txt", "scripts/decoy_matching/prepare_zinc.py build", log["provenance"]["timestamp"][:10], "inputs/external/zinc20/zinc_pool.tsv",
         str(o[str(POOL)]["bytes"]), "md5=%s sha256=%s" % (o[str(POOL)]["md5"], o[str(POOL)]["sha256"]), ">=300000 molecules", str(log["n_pool"]),
         "OK" if log["meets_min_required"] else "FAILED", "unmatched drug-like in-stock pool; schema zinc_pool.schema.json; counts per step results/decoy_matching/zinc_pool_build_log.json"],
        ["D5_zinc_raw_to_pool_map.tsv.gz", "derived from D5_zinc20_2d_tranche_txt", "scripts/decoy_matching/prepare_zinc.py build", log["provenance"]["timestamp"][:10], "inputs/external/zinc20/zinc_raw_to_pool_map.tsv.gz",
         str(o[str(MAPGZ)]["bytes"]), "md5=%s" % o[str(MAPGZ)]["md5"], "one row per downloaded record", str(o[str(MAPGZ)]["n_rows"]), "OK",
         "raw SMILES <-> ZINC id <-> fate (kept / drop reason / duplicate_of)"],
    ]
    new = [r for r in rows if r[0] not in exist]
    for r in new:
        assert len(r) == len(hdr), (len(r), len(hdr))
    with open(MANIFEST, "a") as f:
        for r in new:
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    print("manifest: appended %d rows (%d already present); backup %s" % (len(new), len(rows) - len(new), MANIFEST_BAK))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["list", "plan", "download", "build", "manifest", "all"])
    ap.add_argument("--workers", type=int, default=4, help="parallel HTTP connections (<=16 allowed; 4 default, server runs mod_qos)")
    ap.add_argument("--procs", type=int, default=16, help="RDKit worker processes")
    ap.add_argument("--target-raw-rows", type=int, default=450000)
    ap.add_argument("--max-round", type=int, default=8)
    a = ap.parse_args()
    assert a.workers <= 16 and a.procs <= 16
    st = ["list", "plan", "download", "build", "manifest"] if a.stage == "all" else [a.stage]
    fn = dict(list=stage_list, plan=stage_plan, download=stage_download, build=stage_build, manifest=stage_manifest)
    for s in st:
        print("[%s] === stage %s" % (now(), s), flush=True)
        fn[s](a)


if __name__ == "__main__":
    main()
