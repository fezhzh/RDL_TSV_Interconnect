# v09 RDL LHS Dataset Comparison

This version compares RDL single-device model accuracy for five training data
sets:

- `lhs100`
- `lhs200`
- `lhs400`
- `lhs800`
- `lhs100_lhs200_lhs400_lhs800`

The workflow is:

1. Extract circuit parameters from RDL Snp files.
2. Train MATLAB `input -> 20 -> 20 -> 1` parameter networks for each circuit
   parameter.
3. Use the MATLAB-exported networks as the initial model and fine-tune with
   complex S-parameter loss.

All scripts are direct-run entry points for VS Code.

## Code

| Script | Purpose |
| --- | --- |
| `code/extract_rdl_params_for_lhs_dataset_comparison.py` | Extracts unique RDL circuit parameters from `HFSS_sim/LHS100`, `LHS200`, `LHS400`, and `LHS800`, then writes five dataset CSV groups. |
| `code/nn_train_3.m` | MATLAB training entry for the five datasets. |
| `code/train_matlab_style_rdl_param_nns.py` | Python fallback used in this run because MATLAB CLI crashed before executing `nn_train_3.m`; exports MATLAB-compatible `.mat` parameter networks. |
| `code/finetune_matlab_rdl_models_on_sparams.py` | Loads the `.mat` parameter networks, evaluates initial S-parameter accuracy, then fine-tunes with complex S-parameter loss. |
| `code/plot_lhs_dataset_model_comparison.py` | Plots the five-dataset accuracy comparison from `summary_metrics.csv`; saves figures to `results/dataset_model_comparison_plots/` and shows them when run from VS Code. |
| `code/plot_lhs100_test_model_curve_comparison.py` | Uses `LHS100/test` samples to plot HFSS vs five fine-tuned model curves on the same axes; saves figures to `results/lhs100_test_model_curve_comparison/` and shows them when run from VS Code. |
| `code/compare_lhs800_tsv_dut700_rlgc.py` | Extracts RLGC from `HFSS_sim/LHS800/train/TSV/dut700.s2p` and `dut_700.s2p`, then writes CSV, JSON, plots, and a verification record to `results/rlgc_compare_dut700/`. |

## Results

| Path | Content |
| --- | --- |
| `results/extracted_params/` | Extracted circuit-parameter CSV datasets. |
| `models/matlab_param_nns/` | 90 MATLAB-compatible `.mat` parameter models, plus `matlab_training_summary.csv`. |
| `results/sparam_finetuned_models/` | 10 S-parameter-finetuned `.pt` models, before/after metrics, and 50 worst-test-sample comparison plots. |
| `results/dataset_model_comparison_plots/` | Summary comparison plots for the five training data sets. |
| `results/lhs100_test_model_curve_comparison/` | Curve-level HFSS vs five-model comparison plots on selected LHS100/test samples. |
| `results/rlgc_compare_dut700/` | RLGC extraction and comparison outputs for the LHS800 TSV `dut700.s2p` vs `dut_700.s2p` check, including archived verification text. |

Test-set S-parameter accuracy after S-parameter fine-tuning:

| Dataset | Device | S-MSE | S11 MAE dB | S21 MAE dB |
| --- | --- | ---: | ---: | ---: |
| `lhs100` | TMRDL | 1.220e-3 | 1.493 | 0.0326 |
| `lhs100` | BSMRDL | 1.262e-3 | 1.895 | 0.0390 |
| `lhs200` | TMRDL | 8.87e-4 | 1.671 | 0.0354 |
| `lhs200` | BSMRDL | 4.4e-5 | 0.457 | 0.0103 |
| `lhs400` | TMRDL | 2.9e-5 | 0.240 | 0.0094 |
| `lhs400` | BSMRDL | 3.0e-5 | 0.249 | 0.0062 |
| `lhs800` | TMRDL | 1.3e-5 | 0.160 | 0.0057 |
| `lhs800` | BSMRDL | 5.9e-5 | 0.424 | 0.0072 |
| `lhs100_lhs200_lhs400_lhs800` | TMRDL | 1.1e-5 | 0.151 | 0.0046 |
| `lhs100_lhs200_lhs400_lhs800` | BSMRDL | 6.0e-6 | 0.253 | 0.0036 |
