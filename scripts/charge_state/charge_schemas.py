
M5 = ["naive", "dimorphite_dom", "openbabel", "enum_popmax", "ml_pka"]
CLS = ["<=-2", "-1", "0", "zwitterion", "+1", ">=+2"]
CLS_TAG = {"<=-2": "le_m2", "-1": "m1", "0": "0", "zwitterion": "zw", "+1": "p1", ">=+2": "ge_p2"}

ROW_GRAIN = {
    "charge_per_ligand.csv": "per-ligand (one row per DUD-E active or PDBbind co-crystal ligand)",
    "agreement_pairwise.csv": "per-dataset-per-method-pair (2 datasets x 10 pairs of the 5 methods)",
    "agreement_set_metrics.csv": "per-dataset-per-metric (set-type and neutral-fraction metrics of single methods)",
    "method_class_counts.csv": "per-dataset-method-class (counts of the pH-7.4 state class assigned by a method)",
    "agreement_original_numbers.csv": "per-original-claim (one row per charge-state percentage quoted in the manuscript, with its agreement-wording counterpart)",
    "charge_confusion": "per-row-class (rows: class assigned by method A; columns: class assigned by method B; cells: ligand counts)",
}

PER_LIGAND = [
    ("dataset", "", "dude (3,584 DUD-E actives) or pdbbind (192 co-crystal ligands)"),
    ("compound_id", "", "compound id of the original ledger tables the join key to the original analysis"),
    ("parent_smiles", "", "input structure of the original analysis (the structure every method was run on)"),
    ("q_naive", "e", "net formal charge of the default state = the ensemble member with the smallest |charge| (original 'single'; pdbbind: the faithful ledger choice)"),
    ("q_dimorphite_dom", "e", "net charge of the Dimorphite-DL 2.0.2 dominant state, protonate(ph 7.4, precision 0)"),
    ("q_openbabel", "e", "net charge assigned by OpenBabel 3.1.0 obabel -p 7.4; EMPTY when RDKit cannot sanitise the OpenBabel output (an invalid structure, usually an [O-] with two heavy-atom neighbours; baseline convention)"),
    ("q_openbabel_raw", "e", "net charge of the OpenBabel output read from the unsanitised structure; equals q_openbabel whenever ob_parse_ok is True; sensitivity only, the charge of an invalid structure is not trusted"),
    ("ob_parse_ok", "", "True if RDKit sanitises the OpenBabel output"),
    ("q_enum_popmax", "e", "net charge of the most populated member of the pH 6.4-8.4 Dimorphite ensemble (independent-site Henderson-Hasselbalch populations, original code)"),
    ("q_ml_pka", "e", "net charge of the independent-site Henderson-Hasselbalch dominant state built from MolGpKa pKa predictions (empty = MolGpKa failed)"),
    ("class_naive", "", "charge class of the naive state (<=-2, -1, 0, zwitterion, +1, >=+2); charge-only when the state's SMILES is unavailable"),
    ("class_dimorphite_dom", "", "charge class of the Dimorphite dominant state"),
    ("class_openbabel", "", "charge class of the OpenBabel state"),
    ("class_enum_popmax", "", "charge class of the most populated ensemble member"),
    ("class_ml_pka", "", "charge class of the MolGpKa-based state (zwitterion if the state has a free positive and a free negative charge)"),
    ("dd_state_charges", "e", "net charges of all Dimorphite states at pH 7.4 (default precision), separated by |"),
    ("dd_plausible_charged", "", "True if any Dimorphite state at pH 7.4 is charged (the original '77%' set-type metric)"),
    ("dd_neutral_excluded", "", "True if the naive charge is not among the Dimorphite states at pH 7.4 (the original '9%' set-type metric)"),
    ("ml_q0", "e", "net charge of the MolGpKa input after RDKit uncharging"),
    ("ml_n_base_groups", "count", "MolGpKa basic groups (after merging N-C-N pairs)"),
    ("ml_n_acid_groups", "count", "MolGpKa acidic groups (after merging N-C-N pairs)"),
    ("complete_case", "", "True if all five methods produced a charge (the Fleiss kappa population)"),
]

PAIRWISE = [
    ("dataset", "", "dude or pdbbind"),
    ("method_a", "", "first method of the pair"),
    ("method_b", "", "second method of the pair"),
    ("n", "ligands", "ligands for which both methods produced a charge"),
    ("n_agree", "ligands", "ligands with identical net charge"),
    ("agree_rate", "fraction", "n_agree / n"),
    ("agree_lo", "fraction", "lower 95% Wilson limit of agree_rate"),
    ("agree_hi", "fraction", "upper 95% Wilson limit of agree_rate"),
    ("kappa", "", "Cohen's kappa on the net charge (categories = observed integer charges)"),
    ("kappa_lo", "", "2.5th percentile of kappa over 10,000 bootstrap resamples of ligands"),
    ("kappa_hi", "", "97.5th percentile of kappa over 10,000 bootstrap resamples of ligands"),
    ("kappa_class", "", "Cohen's kappa on the six charge classes (zwitterions separate)"),
    ("kappa_class_lo", "", "2.5th percentile of kappa_class over the same resamples"),
    ("kappa_class_hi", "", "97.5th percentile of kappa_class over the same resamples"),
]

SET_METRICS = [
    ("dataset", "", "dude or pdbbind"),
    ("metric", "", "metric name"),
    ("definition", "", "what the metric counts"),
    ("k", "ligands", "ligands meeting the definition"),
    ("n", "ligands", "ligands in the denominator"),
    ("rate", "fraction", "k / n"),
    ("wilson_lo", "fraction", "lower 95% Wilson limit"),
    ("wilson_hi", "fraction", "upper 95% Wilson limit"),
]

CLASS_COUNTS = [
    ("dataset", "", "dude or pdbbind"),
    ("method", "", "charge-state method"),
    ("charge_class", "", "class of the assigned pH-7.4 state"),
    ("count", "ligands", "ligands assigned to this class by this method"),
    ("fraction", "fraction", "count / ligands with a charge from this method"),
]

OLD_MAP = [
    ("item_id", "", "identifier of the original claim"),
    ("dataset", "", "dude or pdbbind"),
    ("original_claim", "", "the charge-state percentage as quoted in the manuscript (registry token ids in the note)"),
    ("original_metric", "", "name of the field of the original analysis that carries the number"),
    ("original_value", "fraction", "value stored by the original analysis / baseline reproduction"),
    ("reproduced_value", "fraction", "the same metric recomputed in charge-state from the same structures"),
    ("reproduced_ok", "", "True if reproduced_value equals original_value at four decimals"),
    ("agreement_definition", "", "the agreement-wording counterpart: which methods agree on what"),
    ("agreement_value", "fraction", "value of the counterpart (a proportion of ligands)"),
    ("agreement_lo", "fraction", "lower 95% Wilson limit of agreement_value"),
    ("agreement_hi", "fraction", "upper 95% Wilson limit of agreement_value"),
    ("note", "", "registry token ids and wording remarks"),
]

CONFUSION = [("class_a", "", "charge class assigned by method A (row)")] + [("b_" + CLS_TAG[c], "ligands", "ligands assigned class %s by method B" % c) for c in CLS]


ROW_GRAIN.update({
    "anchor_set.csv": "per-molecule (one row per drug-like neutral molecule with a measured, typed pKa ladder; molecules are keyed by the first InChIKey block)",
    "anchor_constants.csv": "per-constant (one row per macroscopic pKa constant of an anchor molecule after merging replicate measurements)",
    "anchor_method_error.csv": "per-method-per-set (5 methods x anchor sets)",
    "anchor_error_by_class.csv": "per-method-per-set-per-functional-class (empty classes are listed with n = 0)",
    "dude_reweighted_error.csv": "per-scheme-anchor-set-method (expected error under the DUD-E stratum composition, with the weights in class_weights)",
    "dude_class_composition.csv": "per-scheme-stratum (DUD-E actives by primary functional class, and by number of predicted ionisable groups [post hoc])",
})

ANCHOR = [
    ("mol_id", "", "running id A0001.. in order of InChIKey skeleton"),
    ("smiles", "", "largest fragment, neutralised, RDKit canonical SMILES (the structure all methods were run on)"),
    ("inchikey", "", "standard InChIKey of smiles"),
    ("ik_skeleton", "", "first block of the InChIKey (14 characters), the key used for de-duplication and for merging sources"),
    ("source", "", "source whose constants are used: IUPAC, SAMPL or ChEMBL (priority order of the config)"),
    ("sources_also", "", "other sources that also hold a usable ladder of this molecule, separated by |"),
    ("sources_concordant", "", "True/False when sources_also is non-empty: the other sources give the same experimental dominant charge; empty otherwise"),
    ("mw", "g/mol", "molecular weight of smiles"),
    ("heavy_atoms", "count", "heavy-atom count of smiles"),
    ("n_sites_dd", "count", "Dimorphite SMARTS ionisable groups with table-mean pKa inside the gate window (N-C-N-linked sites counted once)"),
    ("n_sites_ind", "count", "independent-SMARTS ionisable groups whose nominal pKa range overlaps the gate window (N-C-N-linked sites counted once)"),
    ("n_sites_pred", "count", "max(n_sites_dd, n_sites_ind): predicted ionisable sites required by the strict completeness gate"),
    ("n_sites_rel", "count", "max over the two SMARTS sets of the sites predicted within pH 7.4 +/- relevant_halfwidth: requirement of the relaxed gate"),
    ("n_pka_exp", "count", "macroscopic constants of the molecule after merging replicates"),
    ("n_pka_exp_window", "count", "of these, constants inside the gate window (pH 2-12)"),
    ("n_pka_exp_relevant", "count", "of these, constants within pH 7.4 +/- relevant_halfwidth"),
    ("complete_profile", "", "strict gate: True if n_pka_exp_window >= n_sites_pred, no constant was dropped for dispersion and the ladder has no gap"),
    ("complete_relaxed", "", "relaxed gate (superset of complete_profile): complete_profile, or n_pka_exp_relevant >= n_sites_rel with no dropped constant and no ladder gap"),
    ("had_dropped_constant", "", "True if a constant of this molecule was dropped because its replicate measurements spread by more than the dispersion limit"),
    ("ladder_gap", "", "True if the step numbers of a direction are not 1..k without a gap, or the constants of a direction are not monotone in the step number"),
    ("exp_dominant_charge", "e", "net charge of the most populated charge state at pH 7.4, independent-site Henderson-Hasselbalch on the measured constants"),
    ("exp_pop_max", "fraction", "population of that charge state"),
    ("ambiguous", "", "True if exp_pop_max is below the ambiguity limit"),
    ("coupled_sites", "", "True if two adjacent measured constants inside the gate window lie closer than the coupling gap"),
    ("exp_ladder_charge", "e", "dominant net charge from the exact macroscopic ladder (cross-check of the independent-site result)"),
    ("exp_ladder_agrees", "", "True if exp_ladder_charge equals exp_dominant_charge"),
    ("functional_class", "", "primary functional class of the independent SMARTS set (priority order of the config; none = no site)"),
    ("classes_all", "", "all classes with a site, separated by |"),
    ("q_naive", "e", "net charge of the default state (min |charge| member of the pH 6.4-8.4 Dimorphite ensemble)"),
    ("q_dimorphite_dom", "e", "net charge of the Dimorphite-DL dominant state at pH 7.4"),
    ("q_openbabel", "e", "net charge of the OpenBabel -p 7.4 state; EMPTY when RDKit cannot sanitise the OpenBabel output (baseline convention)"),
    ("q_enum_popmax", "e", "net charge of the most populated pH 6.4-8.4 ensemble member"),
    ("q_ml_pka", "e", "net charge of the independent-site dominant state from MolGpKa predictions"),
    ("ml_in_molgpka_example_sdf", "", "True if the InChIKey skeleton occurs in the 19,848-molecule mols.sdf shipped with MolGpKa"),
    ("ml_in_pkasolver_train", "", "True if the InChIKey skeleton occurs in the pkasolver/Baltruschat training collection"),
    ("ml_in_chembl37", "", "True if the InChIKey skeleton occurs in ChEMBL 37 (upper bound of the unpublished ChEMBL-derived MolGpKa training molecules)"),
]

CONSTANTS = [
    ("mol_id", "", "anchor molecule"),
    ("source", "", "source of the constant"),
    ("direction", "", "AH = conjugate-acid step towards positive charge (pKaH); A = acid step towards negative charge"),
    ("step", "", "step number inside the direction (1 = closest to the neutral molecule)"),
    ("pka", "", "median of the replicate measurements at 20-30 C (pKb converted with pKaH = 14 - pKb)"),
    ("n_records", "count", "records merged into this constant"),
    ("spread", "pKa units", "max - min of the merged records"),
    ("in_window", "", "True if pka is inside the gate window"),
]

METHOD_ERR = [
    ("set", "", "anchor set: complete, complete_relaxed, sensitivity, all, or one of those with a suffix (:excl_ambiguous, :excl_coupled, :IUPAC, :SAMPL, :ChEMBL, :n_sites_1 / :n_sites_ge2 [post hoc], :non_overlap_*)"),
    ("method", "", "naive (pipeline default), neutral_input (input structure as given, net 0), dimorphite_dom, openbabel, enum_popmax or ml_pka"),
    ("k_wrong", "molecules", "molecules whose method net charge differs from the experimental dominant charge"),
    ("n", "molecules", "molecules of the set for which the method produced a charge"),
    ("error_rate", "fraction", "k_wrong / n"),
    ("wilson_lo", "fraction", "lower 95% Wilson limit"),
    ("wilson_hi", "fraction", "upper 95% Wilson limit"),
    ("n_set", "molecules", "molecules in the set"),
    ("n_failed", "molecules", "molecules of the set for which the method produced no charge"),
]

BY_CLASS = [
    ("set", "", "anchor set"),
    ("method", "", "charge-state method"),
    ("functional_class", "", "primary functional class"),
    ("k_wrong", "molecules", "molecules whose method net charge differs from the experimental dominant charge"),
    ("n", "molecules", "molecules of this class and set for which the method produced a charge (0 = empty class)"),
    ("error_rate", "fraction", "k_wrong / n (empty when n = 0)"),
    ("wilson_lo", "fraction", "lower 95% Wilson limit"),
    ("wilson_hi", "fraction", "upper 95% Wilson limit"),
    ("low_n", "", "True if n is below the minimum for a class-level statement"),
]

DUDE_CLASS = [
    ("scheme", "", "stratification: functional_class (as pre-specified) or n_sites_stratum[post_hoc] (added after the first error table was seen)"),
    ("stratum", "", "functional class, or '<=1_group' / '>=2_groups' predicted ionisable groups in the pH 2-12 window"),
    ("n_dude", "ligands", "DUD-E actives in this stratum"),
    ("weight", "fraction", "n_dude / 3,584"),
    ("n_anchor_complete", "molecules", "molecules of this stratum in the complete-profile anchor set"),
    ("covered", "", "True if the stratum has at least one complete-profile anchor molecule"),
]

REWEIGHT = [
    ("scheme", "", "stratification: functional_class (as pre-specified) or n_sites_stratum[post_hoc]"),
    ("anchor_set", "", "anchor set the stratum-level error rates come from"),
    ("method", "", "charge-state method"),
    ("expected_error", "fraction", "sum over covered classes of weight * class error rate, weights renormalised to the covered classes"),
    ("lo", "fraction", "2.5th percentile of the bootstrap (anchor molecules resampled within class, DUD-E composition resampled multinomially)"),
    ("hi", "fraction", "97.5th percentile of the same bootstrap"),
    ("covered_weight", "fraction", "share of DUD-E ligands in classes that have anchor molecules"),
    ("unweighted_error", "fraction", "the plain anchor error rate of the same set, for comparison"),
    ("class_weights", "", "DUD-E weights of the covered strata, stratum:weight separated by |"),
    ("label", "", "extrapolation label: this is a weighted extrapolation, not a measurement on DUD-E"),
]

ANCH_CONF = [
    ("set", "", "anchor set (same names as anchor_method_error.csv)"),
    ("method", "", "charge-state method"),
    ("exp_charge_class", "", "experimental dominant net charge class: <=-2, -1, 0, +1, >=+2"),
    ("method_charge_class", "", "net charge class assigned by the method (same five classes)"),
    ("count", "molecules", "molecules in this cell"),
]
ROW_GRAIN["anchor_confusion.csv"] = "per-set-method-experimental-class-method-class (long-format confusion matrices, zero cells omitted)"
