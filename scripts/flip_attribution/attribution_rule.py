import re
from functools import lru_cache

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
_STEREO = re.compile(r"/(b|t|m|s)[^/]*")


@lru_cache(maxsize=None)
def flat_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


@lru_cache(maxsize=None)
def fixedh_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(Chem.AddHs(m, addCoords=False), options="/FixedH") if m is not None else None


def strip_inchi_stereo(inchi):
    if not isinstance(inchi, str):
        return None
    head, sep, tail = inchi.partition("/f")
    return _STEREO.sub("", head) + sep + _STEREO.sub("", tail)


def change_type(q1, smi1, q2, smi2, inchi1=None, inchi2=None):
    if int(q1) != int(q2):
        return "protonation"
    i1 = strip_inchi_stereo(inchi1 if inchi1 else fixedh_from_smiles(smi1))
    i2 = strip_inchi_stereo(inchi2 if inchi2 else fixedh_from_smiles(smi2))
    if i1 is None or i2 is None:
        return "unparsed"
    if i1 != i2:
        return "tautomer"
    f1, f2 = flat_smiles(smi1), flat_smiles(smi2)
    if f1 != f2:
        return "resonance"
    return "stereo" if smi1 != smi2 else "same_state"

@lru_cache(maxsize=None)
def n_charged_atoms(smiles):
    m = Chem.MolFromSmiles(smiles)
    return -1 if m is None else sum(1 for a in m.GetAtoms() if a.GetFormalCharge() != 0)


def change_type_fine(q1, smi1, q2, smi2, coarse):
    if coarse != "tautomer":
        return coarse
    return "tautomer_ionisation_shift" if n_charged_atoms(smi1) != n_charged_atoms(smi2) else "tautomer_proton_shift"
