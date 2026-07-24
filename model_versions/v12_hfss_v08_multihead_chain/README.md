# v12 HFSS Equivalent-Circuit V08 Multi-Head Chain

This version follows the `v12 model` section in `建模流程.md` and keeps the
entrypoint runnable directly from VS Code without command-line arguments.

## Model Scope

- Full structure: `TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL-TSV-BSMRDL-TSV-TMRDL`.
- Connection count: `12`.
- Connection circuit: the v08 7-parameter Appendix-1 pi circuit
  (`Cn1/Rn1/Cn2/Rn2/Cn3/Rn3/Ln1`).
- Single-device source: HFSS-derived equivalent-circuit models. The current
  RDL backend uses the re-extracted `LHS400_Connection2/train/RDL` checkpoint;
  TMRDL and BSMRDL are mapped into this generic RDL model through their own
  `pitch/l/w/h` feature columns.
- NN flow: a shared 7-parameter pi-network MLP is trained first, then expanded
  into a multi-head model with twelve output heads, one for each RDL/TSV
  junction.
- S-parameter objective: `S11`/`S21` magnitude plus wrapped phase.

## Entry Points

| Path | Purpose |
| --- | --- |
| `code/extract_rdl_connection2_params.py` | Direct VS Code extraction entry. Extracts generic RDL equivalent-circuit parameters from `HFSS_sim/LHS400_Connection2/train/RDL` and writes a version-local parameter CSV. |
| `code/train_rdl_connection2_sparam_model.py` | Direct VS Code training entry. Trains the generic RDL Connection2 model from the extracted RDL CSV and fine-tunes it against the same HFSS S-parameters. |
| `code/extract_tsv_connection2_params.py` | Direct VS Code extraction entry. Extracts TSV equivalent-circuit parameters from `HFSS_sim/LHS400_Connection2/train/TSV` and writes a version-local parameter CSV without overwriting `training_datasets/TSV_TD_4.csv`. |
| `code/train_tsv_connection2_sparam_model.py` | Direct VS Code training entry. Trains a new TSV Connection2 single-device model from the extracted parameter CSV and fine-tunes it against the same HFSS S-parameters. |
| `code/continue_tsv_connection2_sparam_model.py` | Direct VS Code continuation entry. Loads the trained TSV Connection2 checkpoint and continues training with the complex S-parameter objective only, writing before/after metrics and comparison plots to a separate archive. |
| `code/train_v12_hfss_v08_symmetric_multihead.py` | Direct VS Code entry. Builds the v12 long-chain base cascade from HFSS-derived single-device equivalent-circuit models, optimizes one shared v08 circuit per sample, trains the shared parameter network, expands it to twelve output heads, then fine-tunes against `S11/S21` magnitude and wrapped phase. |
| `code/continue_v12_all_train_sparam.py` | Direct VS Code continuation entry. Loads the new-RDL/new-TSV v12 cascade checkpoint, uses all original 150 train samples with no optimization-quality filter, and continues training with the S-parameter objective only. |
| `code/continue_v12_all_train_sparam_round2.py` | Direct VS Code continuation entry. Loads the first all-150 continuation checkpoint and runs a second lower-learning-rate S-parameter-only continuation pass. |
| `code/continue_v12_all_train_sparam_round3.py` | Direct VS Code continuation entry. Loads the second all-150 continuation checkpoint and runs a third lower-learning-rate S-parameter-only continuation pass. |
| `code/continue_v12_all_train_sparam_round4.py` | Direct VS Code continuation entry. Loads the third all-150 continuation checkpoint and runs a fourth lower-learning-rate S-parameter-only continuation pass for overfit checking. |
| `code/recompute_v12_paper_nmse.py` | Direct VS Code report entry. Recomputes the final v12 errors in the paper-style NMSE convention on linear `Re/Im(S11,S21)` curves and writes decimal/percent summaries. |
| `code/recompute_single_device_paper_nmse.py` | Direct VS Code report entry. Recomputes the current v12 RDL and TSV single-device errors in the paper-style NMSE convention on linear `Re/Im(S11,S21)` curves. |
| `code/export_v12_best5_sparams_csv.py` | Direct VS Code export entry. Selects the five best current-best round3 test samples by model NMSE and exports wide CSV tables with one frequency column set and `S11/S21` real/imag columns for HFSS simulation, direct cascade, and cascade model. |
| `code/plot_v12_single_device_model_vs_hfss.py` | Direct VS Code plotting entry. Evaluates TMRDL, BSMRDL, and TSV single-device models against `HFSS_sim/LHS400/train` HFSS S-parameters, then writes metrics, summary CSVs, and random/worst-sample comparison plots. |
| `code/plot_v12_tsv_connection2_model_vs_hfss.py` | Direct VS Code plotting entry for the v12 TSV model against `HFSS_sim/LHS400_Connection2/train/TSV`, using `[r_tsv, h_tsv, pitch]` directly as TSV model input. |

## Inputs

| Data | Path |
| --- | --- |
| RDL/TSV single-device HFSS training source | `HFSS_sim/LHS400_Connection2/train/` |
| Full-chain train target | `HFSS_sim/LHS150_50_Connection2/train/TSV_RDL` |
| Full-chain test target | `HFSS_sim/LHS150_50_Connection2/test/TSV_RDL` |
| RDL equivalent-circuit NN source | `model_versions/v12_hfss_v08_multihead_chain/results/rdl_connection2_sparam_model/rdl_connection2_sparam_net.pt` |
| TSV equivalent-circuit NN source | `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_sparam_continue/tsv_connection2_sparam_continue_net.pt` |

## Outputs

Current best selected model is the round3 all-150 S-parameter continuation
checkpoint. A stable copy and manifest are kept at:

`model_versions/v12_hfss_v08_multihead_chain/results/current_best_model/`

The round4 continuation is retained as an overfit-check archive only; it is not
the current recommended model because its test paper-style NMSE mean is worse
than round3.

Current training script output label is:

`model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv/`

The existing complete archived training output used by the paper draft is:

`model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2/`

The run writes CSV metrics, checkpoints, comparison plots, `training_report.json`,
and `validation_archive.md`.

Single-device comparison output is:

`model_versions/v12_hfss_v08_multihead_chain/results/single_device_model_vs_hfss_lhs400/`

TSV Connection2 single-device comparison output is written by the TSV diagnostic
entry when that script is run:

`model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_model_vs_hfss/`

TSV Connection2 re-extraction and retraining outputs are:

- `model_versions/v12_hfss_v08_multihead_chain/results/rdl_connection2_extracted_params/`
- `model_versions/v12_hfss_v08_multihead_chain/results/rdl_connection2_sparam_model/`
- `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_extracted_params/`
- `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_sparam_model/`
- `model_versions/v12_hfss_v08_multihead_chain/results/tsv_connection2_sparam_continue/`
- `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue/`
- `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round2/`
- `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round3/`
- `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_all150_sparam_continue_round4/`
- `model_versions/v12_hfss_v08_multihead_chain/results/current_best_model/`
- `model_versions/v12_hfss_v08_multihead_chain/results/hfss_v08_symmetric_multihead_lhs150_50_connection2_new_rdl_tsv/paper_nmse_recalculation/`
- `model_versions/v12_hfss_v08_multihead_chain/results/single_device_paper_nmse_recalculation/`
- `model_versions/v12_hfss_v08_multihead_chain/results/best5_sparameter_csv_current_best_round3/`

## Documents

| Path | Content |
| --- | --- |
| `v12建模流程小论文.md` | English Markdown paper draft revised for the v12 HFSS-equivalent-circuit backend, v08 pi connection network, shared optimization, twelve-head multi-head flow, and TSV-RDL3 validation case metrics. |
| `v12建模流程小论文.docx` | English Word small-paper version generated from the Markdown draft using `Manuscript.doc` as the style/template source; missing method figures are kept as blank placeholders with captions. |
| `code/build_v12_paper_docx.py` | Direct VS Code build entry for regenerating the English Word paper from the v12 metrics, template conversion, local assets, and placeholder figure definitions. |
| `results/v12_paper_docx_validation_archive.md` | Validation archive for the Word paper generation, English revision, metric sources, PDF/PNG render, and visual QA result. |
| `results/paper_method_revision_validation_archive.md` | Local validation archive for the v12 paper method revision and README index updates. |
