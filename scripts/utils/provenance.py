import csv, hashlib, importlib, json, platform, socket, sys, time
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _delim(path):
    return "\t" if str(path).endswith(".tsv") else ","


def count_records(path):
    p = str(path)
    if p.endswith((".csv", ".tsv")):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in csv.reader(f, delimiter=_delim(p)))
        return max(n - 1, 0), "table_rows"
    if p.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                o = json.load(f)
            if isinstance(o, dict):
                return len(o), "json_top_level_keys"
            if isinstance(o, list):
                return len(o), "json_list_items"
        except Exception:
            pass
    with open(path, "rb") as f:
        return sum(1 for _ in f), "lines"


def input_record(path):
    p = Path(path)
    if not p.is_file():
        return dict(path=str(p), exists=False)
    n, unit = count_records(p)
    return dict(path=str(p), exists=True, bytes=p.stat().st_size, md5=file_md5(p), n_records=n, record_unit=unit)


def tool_versions(extra=()):
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "pandas", "yaml") + tuple(extra):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception:
            out[m] = "not-installed"
    return out


def provenance_block(script, config=None, inputs=(), seed=None, n_boot=None, n_perm=None, extra=None, tools_extra=()):
    s = Path(script)
    blk = dict(script=str(s), script_md5=file_md5(s) if s.is_file() else None,
               config=str(config) if config else None,
               config_md5=file_md5(config) if config and Path(config).is_file() else None,
               inputs=[input_record(p) for p in inputs],
               seed=seed, n_boot=n_boot, n_perm=n_perm,
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               host=socket.gethostname(), tool_versions=tool_versions(tools_extra))
    if extra:
        blk["extra"] = extra
    return blk


def _infer_type(vals):
    vals = [v for v in vals if v != ""]
    if not vals:
        return "empty"

    def ok(fn, s):
        try:
            fn(s)
            return True
        except ValueError:
            return False
    if all(ok(int, v) for v in vals):
        return "int"
    if all(ok(float, v) for v in vals):
        return "float"
    return "str"


def _read_table(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f, delimiter=_delim(p))
        header = next(rd)
        rows = list(rd)
    return header, rows


def write_schema(csv_path, columns, row_grain, description, extra=None):
    p = Path(csv_path)
    header, rows = _read_table(p)
    names = [c[0] for c in columns]
    if header != names:
        raise ValueError("%s: declared columns != header | declared-only=%s | header-only=%s | same-set-different-order=%s"
                         % (p.name, [n for n in names if n not in header], [n for n in header if n not in names], sorted(names) == sorted(header)))
    ragged = [i for i, r in enumerate(rows) if len(r) != len(header)]
    if ragged:
        raise ValueError("%s: %d ragged rows (first at data row %d)" % (p.name, len(ragged), ragged[0]))
    cols = []
    for i, (name, unit, desc) in enumerate(columns):
        vals = [r[i] for r in rows]
        cols.append(dict(name=name, type=_infer_type(vals), unit=unit, description=desc, n_missing=sum(1 for v in vals if v == "")))
    sch = dict(file=p.name, description=description, row_grain=row_grain, n_rows=len(rows), n_columns=len(header),
               delimiter="," if _delim(p) == "," else "\\t", file_md5=file_md5(p), columns=cols)
    if extra:
        sch["extra"] = extra
    out = p.with_name(p.stem + ".schema.json")
    out.write_text(json.dumps(sch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def verify_schema(csv_path):
    p = Path(csv_path)
    sp = p.with_name(p.stem + ".schema.json")
    if not sp.is_file():
        return ["missing schema file: " + sp.name]
    s = json.loads(sp.read_text(encoding="utf-8"))
    header, rows = _read_table(p)
    probs = []
    if s.get("file_md5") != file_md5(p):
        probs.append("file_md5 differs from the CSV on disk (schema is stale)")
    if s.get("n_rows") != len(rows):
        probs.append("n_rows %s != %d" % (s.get("n_rows"), len(rows)))
    if [c["name"] for c in s.get("columns", [])] != header:
        probs.append("column names differ from the CSV header")
    for c in s.get("columns", []):
        if not c.get("description"):
            probs.append("column '%s' has no description" % c["name"])
    if s.get("n_columns") != len(header):
        probs.append("n_columns %s != %d" % (s.get("n_columns"), len(header)))
    if s.get("delimiter") != ("," if _delim(p) == "," else "\\t"):
        probs.append("delimiter %r differs from the file" % s.get("delimiter"))
    if all(len(r) == len(header) for r in rows):
        for i, c in enumerate(s.get("columns", [])):
            vals = [r[i] for r in rows]
            if c.get("type") != _infer_type(vals):
                probs.append("column '%s': declared type %s, data say %s" % (c["name"], c.get("type"), _infer_type(vals)))
            if c.get("n_missing") != sum(1 for v in vals if v == ""):
                probs.append("column '%s': declared n_missing %s, data say %d" % (c["name"], c.get("n_missing"), sum(1 for v in vals if v == "")))
    for k in ("row_grain", "description"):
        if not s.get(k):
            probs.append("missing field: " + k)
    return probs


def write_provenance_sidecar(json_path, block, note="attached post hoc; the result file's bytes are unchanged"):
    p = Path(json_path)
    out = p.with_name(p.stem + ".provenance.json")
    out.write_text(json.dumps(dict(result_file=p.name, result_file_md5=file_md5(p), note=note, provenance=block),
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
