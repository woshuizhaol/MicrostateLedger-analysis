import csv, json, subprocess
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "$ANALYSIS_ROOT/inputs/data"
OBABEL = "$ANALYSIS_ROOT/env/conda/bin/obabel"


def fc(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else Chem.GetFormalCharge(m)


def obabel_batch(smiles, ph):
    inp = "\n".join(f"{s}\t{i}" for i, s in enumerate(smiles)) + "\n"
    p = subprocess.run([OBABEL, "-ismi", "-osmi", "-p", str(ph)], input=inp, capture_output=True, text=True)
    out = {}
    for line in p.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                out[int(parts[1].strip())] = fc(parts[0].strip())
            except ValueError:
                pass
    return out


rows = [r for r in csv.DictReader(open(f"{D}/iupac_pka_detail.tsv"), delimiter="\t")
        if r["pred_pka_closest"]]
smiles = [r["smiles"] for r in rows]
exp_pka = [float(r["exp_pka"]) for r in rows]
pred_pka = [float(r["pred_pka_closest"]) for r in rows]

lo = obabel_batch(smiles, 1.5)
hi = obabel_batch(smiles, 12.5)
mid = obabel_batch(smiles, 7.4)

n = cheap_err = ob_err = mono = skipped = 0
for i in range(len(rows)):
    cprot, cdeprot, cob = lo.get(i), hi.get(i), mid.get(i)
    if cprot is None or cdeprot is None or cob is None:
        skipped += 1; continue
    if cprot == cdeprot:
        skipped += 1; continue
    if abs(cprot - cdeprot) == 1:
        mono += 1
    n += 1
    exp_dom = cprot if exp_pka[i] > 7.4 else cdeprot
    cheap_dom = cprot if pred_pka[i] > 7.4 else cdeprot
    if cheap_dom != exp_dom:
        cheap_err += 1
    if cob != exp_dom:
        ob_err += 1

res = dict(dataset="IUPAC_experimental_anchor", n_scored=n, n_skipped_no_transition=skipped,
           n_monoprotic_delta1=mono,
           cheap_pH74_dominant_error_vs_experiment=round(cheap_err / n, 4),
           openbabel_pH74_dominant_error_vs_experiment=round(ob_err / n, 4),
           note="charge states from OpenBabel at pH 1.5/12.5; pH-7.4 side chosen by the method's pKa; "
                "truth uses the EXPERIMENTAL pKa. Cheap error here should match the reported 6.6% flip rate.")
print(json.dumps(res, indent=2))
json.dump(res, open(f"$ANALYSIS_ROOT/results/benchmark_analyses/anchor_iupac/production_protonation_iupac_anchor.json", "w"), indent=2)
print("DONE_ANCHOR")
