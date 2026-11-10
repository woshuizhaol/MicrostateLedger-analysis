import os
import csv, hashlib, json, sys, time, urllib.request
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", ".")); OUT = ROOT / "inputs/external/chembl"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json?standard_type=pKa&limit=1000&offset=%d"
FIELDS = ["activity_id", "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles", "assay_chembl_id", "assay_type", "assay_description", "document_chembl_id", "standard_type", "standard_relation",
          "standard_value", "standard_units", "type", "relation", "value", "units", "data_validity_comment", "activity_comment", "potential_duplicate", "src_id", "target_chembl_id", "bao_label"]
def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}), timeout=120) as r: return r.read()
        except Exception as e:
            print("retry", k, url[-40:], type(e).__name__, str(e)[:80], flush=True); time.sleep(5 * (k + 1))
    raise SystemExit("download failed: " + url)
off, total, pages = 0, None, []
while total is None or off < total:
    p = OUT / ("activity_pKa_offset_%06d.json" % off)
    if not p.exists(): p.write_bytes(get(BASE % off)); time.sleep(0.5)
    d = json.loads(p.read_bytes()); total = d["page_meta"]["total_count"]; pages.append(p); print("page offset", off, "records", len(d["activities"]), "total", total, flush=True); off += 1000
rows = []
for p in pages:
    for a in json.loads(p.read_bytes())["activities"]: rows.append({k: ("" if a.get(k) is None else a.get(k)) for k in FIELDS})
with open(OUT / "chembl_pka_activities.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
print("records", len(rows), "expected", total, "unique activity_id", len({r["activity_id"] for r in rows}))
if len(rows) != total: sys.exit("record count differs from the API total_count")
