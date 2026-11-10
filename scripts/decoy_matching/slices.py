import csv, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get("ANALYSIS_ROOT", "."))
sys.path.insert(0, str(ROOT / "scripts/utils"))
import provenance as P

A = yaml.safe_load(open(ROOT / "configs/decoy_matching_slices.yaml", encoding="utf-8"))["slices"]
OUT = Path(os.environ.get("DECOY_OUT_ROOT", str(ROOT)))
Q = ROOT / "raw_outputs/decoy_matching/queue/ligand_sets.csv"
rows = list(csv.DictReader(open(Q, encoding="utf-8")))
K = int(A["n_slices"]); rng = random.Random(int(A["seed"]))
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[(r["set"], r["target"], r["role"], r["round"])].append(i)
for key in sorted(groups):
    idx = groups[key][:]; rng.shuffle(idx)
    for rank, i in enumerate(idx):
        rows[i]["slice"] = rank * K // len(idx) + 1
keys = sorted({(r["set"], r["target"]) for r in rows}); orng = random.Random(20261023); orng.shuffle(keys); rank_of = {k: i for i, k in enumerate(keys)}
rows.sort(key=lambda r: (r["slice"], rank_of[(r["set"], r["target"])], int(r["round"]), r["ligand_id"], r["smiles"]))
cols = list(rows[0].keys())
outq = OUT / "raw_outputs/decoy_matching/queue/ligand_sets_sliced.csv"
outq.parent.mkdir(parents=True, exist_ok=True)
with open(outq, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
pil = json.load(open(ROOT / "results/decoy_matching/budget_pilot.json"))["pools"]
mean_el = {"dude_full/active": pil["dude_active"]["eligible_mean"], "dude_full/negative": pil["dude_decoy"]["eligible_mean"], "litpcba/active": pil["litpcba_active"]["eligible_mean"],
           "litpcba/negative": pil["litpcba_inactive"]["eligible_mean"], "zinc_unmatched/negative": pil["zinc"]["eligible_mean"], "zinc_charge_mismatch/negative": pil["zinc"]["eligible_mean"]}
per = defaultdict(lambda: Counter())
for r in rows:
    per[int(r["slice"])]["%s/%s" % (r["set"], r["role"])] += 1
plan = {}
for k in sorted(per):
    ligs = sum(per[k].values()); dock = sum(n * mean_el[key] for key, n in per[k].items())
    plan[k] = dict(ligands=ligs, expected_dockings=round(dock), by_set=dict(per[k]))
blk = P.provenance_block(Path(__file__), ROOT / "configs/decoy_matching_slices.yaml", [Q], seed=int(A["seed"]), extra=dict(n_slices=K))
cols_schema = [(c, "", {"set": "ligand set", "target": "receptor target", "role": "active / negative", "round": "0 active, k = k-th negative of an active (unchanged)", "ligand_id": "database id", "smiles": "SMILES", "info": "decoy/inactive or ZINC info",
                        "slice": "processing slice 1..%d (Amendment 1)" % K}[c]) for c in cols]
cs = [(c, u, d) for c, (u, d) in zip(cols, [("", d) for _, _, d in [(c, "", dd) for c, _, dd in cols_schema]])]
P.write_schema(outq, [(c, "", d) for c, _, d in cols_schema], "one row per (set, target, ligand) in docking order", "Docking queue of decoy-matching ordered in random slices (Amendment 1); same rows as ligand_sets.csv", extra=dict(provenance=blk))
res = OUT / "results/decoy_matching"; res.mkdir(parents=True, exist_ok=True)
json.dump(dict(provenance=blk, n_slices=K, plan=plan, total_expected_dockings=sum(v["expected_dockings"] for v in plan.values())), open(res / "slice_plan.json", "w"), indent=2)
print("rows", len(rows), "| slice sizes (ligands):", [plan[k]["ligands"] for k in sorted(plan)], "| expected dockings per slice ~", plan[1]["expected_dockings"], "| total", sum(v["expected_dockings"] for v in plan.values()))
