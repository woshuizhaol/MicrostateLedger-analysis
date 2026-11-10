# Supplementary Software 1

Scripts, configuration files and result tables of the docking, scoring, tautomer-ranking, charge-state and pose-validity analyses reported in "Scoring accuracy reveals ligand microstate errors missed by docking-enrichment benchmarks". The MicrostateLedger framework itself is archived at https://doi.org/10.5281/zenodo.21162805.

## Contents

| Folder | Content |
|---|---|
| `scripts/<analysis>/` | Python scripts of each analysis (see the table below); `scripts/utils` holds the shared helper that writes the provenance block of each result |
| `configs/` | YAML files with the parameters, input locations and random seeds of each analysis |
| `results/<analysis>/` | Result tables (CSV) and summaries (JSON); each table has a `*.schema.json` describing its columns and the script, parameters and input checksums that produced it |
| `env/` | Conda environment and `pip freeze` of the analysis environment and of the pKa-prediction environment |

## Analyses

`ten_target` denotes the docking benchmark on ten DUD-E targets (40 ligands per target, five seeds); `full_scale` denotes the docking of every enumerated microstate of all ligands of the four ligand sets (DUD-E, LIT-PCBA and the two ZINC20 sets); `budget6_docking` is the ten-target benchmark with a budget of six dockings per ligand.

| Folder | Content | Order of the scripts |
|---|---|---|
| `tautomer_ranking` | Tautomer ranking on a common set of molecules | `tautomer`, `rdkit_version_comparison` |
| `charge_state` | Agreement of charge-state methods and the experimental reference set | `download_chembl`, `agreement`, `build_reference_set`, `reference_errors`, `pka_threshold`, `context_numbers` |
| `casf_scoring` | Random-microstate null for CASF-2016 scoring power | `states`, `random_null`, `predictor_contrast`, `reference_coverage` |
| `decoy_matching` | Property comparison of actives and negatives, full-scale docking, tests of the activity independence of the gain | `prepare_zinc`, `properties`, `zinc_properties`, `litpcba_properties`, `build_sets`, `slices`, `dock` (resumable), `property_comparison`, `assemble`, `gain_tests`, `positive_control`, `null_calibration`, `posthoc_sensitivity`, `posthoc_descriptive`, `ten_target_reanalysis` |
| `pose_validity` | PoseBusters validity of poses, flips between microstates, docking noise | `manifest_ten_target`, `posebusters_batch --source ten_target`, `flips_ten_target`, `posebusters_batch --source full_scale`, `flips_full_scale`, `noise_sample`, `noise_dock`, `posebusters_batch --source noise`, `noise_analysis`, `exclusion_sensitivity`, `seed_overlap_noise` |
| `pose_criteria` | Crystal-reference criteria (RMSD, interaction fingerprint) on CASF-2016 | `redock`, `pose_checks`, `criterion_flips`, `score_rank_null` |
| `flip_attribution` | Attribution of validity flips to the kind of microstate change | `attribution --source ten_target`, `--source casf`, `--source full_scale` |
| `benchmark_analyses` | Recomputation of the charge-state and score-swing numbers of the ten-target benchmark | `anchor_iupac`, `protonation_tools`, `swing_signal` |

## Running the analyses

Scripts read three root directories from environment variables: `ANALYSIS_ROOT` (outputs and intermediate files), `DOCKING_ROOT` and `BASE_ROOT` (docking runs and earlier result tables that serve as inputs). Output locations can be redirected with `TAUTOMER_OUT_ROOT`, `CHARGE_OUT_ROOT`, `CASF_OUT_ROOT`, `DECOY_OUT_ROOT`, `POSE_OUT_ROOT`, `CRITERIA_OUT_ROOT` and `ATTR_OUT_ROOT`. The input data (DUD-E, LIT-PCBA, PDBbind, CASF-2016, ZINC20, ChEMBL, the IUPAC dissociation-constant data set and the SAMPL challenge data) are not included; obtain them from their providers and set the paths in `configs/`.

The docking steps (`dock`, `noise_dock`, `redock`) use a separate environment with RDKit 2025.09.5, Meeko 0.7.1, Dimorphite-DL 2.0.2 and AutoDock Vina (build 7ac2999-mod, see `configs/docking_protocol.yaml`); the remaining scripts use `env/environment.yml`. The pKa predictor MolGpKa runs in the environment of `env/environment_pka_ml.yml`.

The tables of the manuscript and the Supplementary Information can be recomputed from the files in `results/` without repeating the docking.
