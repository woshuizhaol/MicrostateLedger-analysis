import csv, json, subprocess, os, sys
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import dimorphite_dl

D = "$ANALYSIS_ROOT/inputs/data"
OUT = "$ANALYSIS_ROOT/results/benchmark_analyses/protonation_tools"
OBABEL = "$ANALYSIS_ROOT/env/conda/bin/obabel"


def formal_charge(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.GetFormalCharge(m)


def openbabel_ph74(smiles_list):
    inp = "\n".join(f"{s}\t{i}" for i, s in enumerate(smiles_list)) + "\n"
    p = subprocess.run([OBABEL, "-ismi", "-osmi", "-p", "7.4"], input=inp,
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        smi, idx = parts[0].strip(), parts[1].strip()
        try:
            out[int(idx)] = formal_charge(smi)
        except ValueError:
            pass
    return out


def dimorphite_charges(smi):
    try:
        states = dimorphite_dl.protonate_smiles(smi, ph_min=7.4, ph_max=7.4)
    except Exception:
        return None
    chs = set()
    for s in states:
        c = formal_charge(s)
        if c is not None:
            chs.add(c)
    return chs or None


def analyze(name, rows, smi_key, default_key):
    smiles = [r[smi_key] for r in rows]
    defaults = [int(r[default_key]) for r in rows]
    ob = openbabel_ph74(smiles)
    n = 0
    ob_ne_default = 0
    ob_nonzero = 0
    dd_has_charge = 0
    dd_ne_default = 0
    both_ne = 0
    for i, (smi, dflt) in enumerate(zip(smiles, defaults)):
        obc = ob.get(i)
        ddc = dimorphite_charges(smi)
        if obc is None and ddc is None:
            continue
        n += 1
        ob_bad = obc is not None and obc != dflt
        if ob_bad:
            ob_ne_default += 1
        if obc is not None and obc != 0:
            ob_nonzero += 1
        dd_bad = False
        if ddc is not None:
            if any(c != 0 for c in ddc):
                dd_has_charge += 1
            if dflt not in ddc:
                dd_ne_default += 1
                dd_bad = True
        if ob_bad and dd_bad:
            both_ne += 1
    res = dict(dataset=name, n=n,
               openbabel_ne_default_frac=round(ob_ne_default / n, 4),
               openbabel_nonzero_frac=round(ob_nonzero / n, 4),
               dimorphite_has_charged_state_frac=round(dd_has_charge / n, 4),
               dimorphite_default_absent_frac=round(dd_ne_default / n, 4),
               both_tools_disagree_frac=round(both_ne / n, 4))
    print(json.dumps(res, indent=2))
    return res


results = {}

dude = list(csv.DictReader(open(f"{D}/dude_single_vs_dominant.tsv"), delimiter="\t"))
results["dude"] = analyze("DUD-E", dude, "parent_smiles", "single_charge")

pdb = list(csv.DictReader(open(f"{D}/pdbbind_faithful_single_vs_dominant.tsv"), delimiter="\t"))
if pdb:
    print("PDBbind cols:", list(pdb[0].keys()))
    smi_k = "parent_smiles" if "parent_smiles" in pdb[0] else [k for k in pdb[0] if "smiles" in k.lower()][0]
    dfl_k = "single_charge" if "single_charge" in pdb[0] else [k for k in pdb[0] if "single" in k.lower() and "charge" in k.lower()][0]
    results["pdbbind"] = analyze("PDBbind", pdb, smi_k, dfl_k)

json.dump(results, open(f"{OUT}/production_protonation_charge.json", "w"), indent=2)
print("SAVED production_protonation_charge.json")
print("DONE_EXP")
