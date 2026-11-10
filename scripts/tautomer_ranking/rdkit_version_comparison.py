import csv, hashlib, json, sys

sys.dont_write_bytecode = True


def rescore_all(pairs_tsv):
    from rdkit import Chem, rdBase, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize as MS
    RDLogger.DisableLog("rdApp.*")
    te = MS.TautomerEnumerator()
    rows, h, hs = [], hashlib.sha256(), hashlib.sha256()
    with open(pairs_tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            m1, m2 = Chem.MolFromSmiles(r["t1"]), Chem.MolFromSmiles(r["t2"])
            if m1 is None or m2 is None:
                raise ValueError("unparsable SMILES in row %s" % r["row"])
            s1, s2 = te.ScoreTautomer(m1), te.ScoreTautomer(m2)
            dom = "T1" if s1 > s2 else ("T2" if s2 > s1 else "tie")
            try:
                can1 = Chem.MolToSmiles(te.Canonicalize(m1))
            except Exception:
                can1 = None
            cdom = "T1" if can1 == r["t1"] else ("T2" if can1 == r["t2"] else "other")
            same = (str(s1) == r["score_t1"] and str(s2) == r["score_t2"] and dom == r["score_dom"] and cdom == r["canon_dom"])
            rows.append(dict(row=r["row"], score_t1=s1, score_t2=s2, score_dom=dom, canon_dom=cdom, matches_stored=same))
            h.update(("%s|%s|%s|%s|%s\n" % (r["row"], s1, s2, dom, cdom)).encode()); hs.update(("%s|%s|%s|%s\n" % (r["row"], s1, s2, dom)).encode())
    return rows, rdBase.rdkitVersion, h.hexdigest(), hs.hexdigest()


if __name__ == "__main__":
    rows, version, digest, score_digest = rescore_all(sys.argv[1])
    print(json.dumps(dict(rdkit_version=version, n_pairs=len(rows), digest_sha256=digest, score_digest_sha256=score_digest, canon_dom={r["row"]: r["canon_dom"] for r in rows},
                          n_matching_stored=sum(1 for r in rows if r["matches_stored"]),
                          n_score_tie=sum(1 for r in rows if r["score_dom"] == "tie"))))
