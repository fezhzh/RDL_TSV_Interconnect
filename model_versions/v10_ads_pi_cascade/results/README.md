# v10 Results

Generated result files are ignored by Git except this README.

Expected outputs from `code/train_ads_pi_cascade_v10.py`:

- `pi_optimization_summary.csv`
- `pi_optimized_targets.csv`
- `pi_connection_net_param_pretrain.pt`
- `pi_sparam_metrics_after_param_pretrain.csv`
- `pi_sparam_summary_after_param_pretrain.csv`
- `pi_param_predictions_after_param_pretrain.csv`
- `pi_training_history.csv`
- `pi_sparam_metrics.csv`
- `pi_sparam_summary.csv`
- `pi_param_predictions.csv`
- `pi_connection_net.pt`
- `training_report.json`
- `validation_archive.md`
- `comparison_plots/`: HFSS simulation vs ADS direct cascade vs optimized pi
  vs pi-NN model plots for `S11` and `S21` real/imaginary components.
  `random_test/` contains fixed-seed random test samples, and `worst_test/`
  contains the worst test samples sorted by final pi-NN NMSE.

The current error metric is NMSE:
`sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`, where `y` is the
flattened `S11.real`, `S11.imag`, `S21.real`, and `S21.imag` vector over all
frequencies. The relevant columns are named `*_nmse_s11_s21_ri_*`.

## ADS Single-Device Calibration

`ads_single_device_calibration_small` calibrates ADS single-device simulations
against a small HFSS subset from `LHS200/train`, DUTs `100` to `103`.

Sweep variables:

- Primary: `er_si`, `cond`, `tand`, and TSV `c1_scale`.
- Secondary: RDL `l_scale`, `w_scale`, `pitch_scale`, `h_tsv_scale`,
  `h_rdl_scale`; TSV `pitch_scale`, `h_tsv_scale`, `d_scale`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10.py`

Best small-subset settings:

| Scope | er_si | cond | tand | c1_scale | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | d_scale | NMSE Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDL | 10.8 | 5.8e7 | 0.005 | - | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 | - | 0.013304 |
| TSV | 11.9 | 5.8e7 | 0.005 | 1.0 | - | - | 1.0 | 1.0 | - | 1.0 | 0.002765 |

On this four-DUT calibration set, lowering RDL `er_si` and `w_scale` improves
RDL fit. TSV baseline remains best; changing `c1_scale` did not improve the
small-subset TSV metric.

## Refined ADS Single-Device Calibration

`ads_single_device_calibration_refined` repeats the single-device calibration
with six HFSS DUTs from `LHS200/train`, DUTs `100` to `105`, and combined
candidate settings around the first calibration result.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_refined.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_refined.py`

Best refined settings:

| Scope | er_si | cond | tand | c1_scale | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | d_scale | NMSE Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDL | 10.2 | 5.8e7 | 0.005 | - | 1.0 | 0.85 | 1.0 | 1.2 | 1.0 | - | 0.008500 |
| TSV | 11.9 | 5.8e7 | 0.005 | 1.0 | - | - | 1.0 | 1.0 | - | 1.0 | 0.002484 |

Compared with `ads_single_device_calibration_small`, the RDL mean NMSE improves
from `0.013304` to `0.008500`. TSV remains best at the baseline settings.

## 16-DUT ADS Single-Device Calibration

`ads_single_device_calibration_16dut` evaluates the same refined candidate
search on 16 evenly spaced DUTs from `LHS200/train`:
`100, 113, 126, 140, 153, 166, 180, 193, 206, 219, 233, 246, 259, 273, 286, 299`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_refined.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_16dut.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_16dut.py`

Best 16-DUT settings:

| Scope | er_si | cond | tand | c1_scale | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | d_scale | NMSE Mean | NMSE Median | NMSE Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDL | 10.2 | 5.8e7 | 0.005 | - | 1.0 | 0.85 | 1.0 | 1.0 | 0.8 | - | 0.006183 | 0.003774 | 0.021594 |
| TSV | 11.9 | 5.8e7 | 0.005 | 1.0 | - | - | 1.0 | 1.1 | - | 1.0 | 0.001169 | 0.000376 | 0.005262 |

Compared with the six-DUT refined calibration, the broader set keeps the RDL
best point near `er_si=10.2` and `w_scale=0.85`, but shifts the height
correction from RDL `h_tsv_scale=1.2` to RDL `h_rdl_scale=0.8`. TSV no longer
selects pure baseline; the best candidate uses `h_tsv_scale=1.1`.

Additional best-setting comparison plots were generated with
`code/plot_ads_single_device_calibration_16dut.py`:

- `plots_all_best/`: 48 plots, covering TMRDL, BSMRDL, and TSV for all 16 DUTs.
- `plots_all_best_summary.csv`: per-plot NMSE and source path summary.
- `plots_all_best_validation.md`: validation archive for the additional plot run.

Mean NMSE by plotted device:

| Scope | Device | Count | NMSE Mean | NMSE Median | NMSE Max |
| --- | --- | ---: | ---: | ---: | ---: |
| RDL | BSMRDL | 16 | 0.008340 | 0.005395 | 0.021594 |
| RDL | TMRDL | 16 | 0.004026 | 0.002507 | 0.012753 |
| TSV | TSV | 16 | 0.001169 | 0.000376 | 0.005262 |

## RDL Netlist Update Calibration

`ads_cal_rdl_update16` recalibrates the RDL ADS settings after the RDL ADS
netlist was updated to the `MLIN2` structure. The same 16-DUT LHS200 subset is
used. This is an RDL-only recalibration; TSV settings are carried over from
`ads_single_device_calibration_16dut`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_16dut_rdl_net_update.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_refined.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py`
- RDL generated netlist check confirmed the updated template still maps
  `l_rdl`, `w_rdl`, `h_rdl`, `er_si`, `cond`, and `tand`.
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_single_devices_v10_16dut_rdl_net_update.py`

Best RDL net-update settings:

| Scope | er_si | cond | tand | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | NMSE Mean | NMSE Median | NMSE Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDL | 10.2 | 5.8e7 | 0.005 | 1.0 | 0.85 | 1.1 | 1.0 | 1.0 | 0.027345 | 0.018205 | 0.068977 |

Top RDL candidates after the netlist update:

| Candidate | NMSE Mean | NMSE Median | NMSE Max |
| --- | ---: | ---: | ---: |
| er10.2_w0.85_pitch1.1 | 0.027345 | 0.018205 | 0.068977 |
| er9.8_w0.8 | 0.029633 | 0.021392 | 0.071747 |
| er10.8_w0.85_pitch1.1 | 0.029841 | 0.022497 | 0.072168 |

Compared with the previous 16-DUT RDL calibration (`0.006183` mean NMSE), the
updated RDL netlist is worse on this LHS200 single-device calibration set.

## LHS400_Connection2 Random-10 RDL Calibration

`ads_cal_rdl_lhs400c2_rand10` restores the original RDL ADS `MCLIN` netlist and
calibrates ADS RDL settings on 10 random samples from
`HFSS_sim/LHS400_Connection2/train/RDL`. The source RDL data use 1000 frequency
points from 0.1 to 100 GHz, and ADS generated netlists use the same sweep.

Random subset:

`12, 45, 116, 150, 187, 242, 269, 298, 335, 373`

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_rdl_lhs400_connection2_random10.py`
- Sample check confirmed 10 random DUTs with seed `20260708`.
- RDL generated netlist check confirmed original `MCLIN:CLin1`, `MSUB H=h_tsv`,
  and `Port:Term1 N__3 N__5`.
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_rdl_lhs400_connection2_random10.py`

Best settings:

| Scope | er_si | cond | tand | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | NMSE Mean | NMSE Median | NMSE Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RDL | 12.5 | 5.8e7 | 0.005 | 1.0 | 0.8 | 1.0 | 1.0 | 1.0 | 0.129976 | 0.139975 | 0.188476 |

Top candidates:

| Candidate | NMSE Mean | NMSE Median | NMSE Max |
| --- | ---: | ---: | ---: |
| er12.5_w0.8 | 0.129976 | 0.139975 | 0.188476 |
| l1.05 | 0.130940 | 0.141183 | 0.187928 |
| er12.5_w0.85 | 0.131931 | 0.141493 | 0.190570 |

The original-MCLIN ADS RDL fit remains poor on this 0.1-100 GHz LHS400
Connection2 subset. The best candidate is only marginally better than nearby
alternatives, indicating this netlist/settings search is not sufficient by
itself for this wider-band RDL dataset.

## Latest Smoke Validation

`ads_pi_cascade_smoke` was run with the `development_cached_snp` backend to
validate the method before ADS paths are configured.

| Split | Count | Direct MSE Mean | Pi-NN MSE Mean | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 6 | 1.73673e-1 | 2.63409e-2 | 4.36124 | 1.32606 |
| train | 18 | 1.45875e-1 | 2.75317e-2 | 3.03604 | 1.60034 |
| val | 6 | 1.60365e-1 | 3.21220e-2 | 3.35537 | 1.67821 |

## ADS Helper Validation

`rdl_ads_sim/ADS_Sim.py` and `tsv_ads_sim/ADS_Sim.py` are importable
single-device simulator helpers. Verification completed:

- `python -m py_compile model_versions/v10_ads_pi_cascade/rdl_ads_sim/ADS_Sim.py model_versions/v10_ads_pi_cascade/tsv_ads_sim/ADS_Sim.py model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10.py`
- `conda run -n PyML python -c "... ads_variables_for_device(...) ..."`

The mapping smoke check produced SI values for the demo structure:

| Device | Checked variable | Value |
| --- | --- | ---: |
| TMRDL | `l_rdl` | `1.0e-4` |
| BSMRDL | `l_rdl` | `1.0e-4` |
| TSV | `d_tsv` | `1.0e-5` |

ADS solver execution has also been checked through the small ADS smoke runs.

## ADS Smoke And Material Checks

The ADS backend was run with 3 train, 3 validation, and 3 test samples. Each
sample uses ADS-simulated `TMRDL`, `BSMRDL`, and `TSV` single-device
S-parameters, then optimizes and trains the pi cascade model.

| Run | ADS setting change | Test Pi-NN MSE | Val Pi-NN MSE | Test S11 MAE dB | Test S21 MAE dB |
| --- | --- | ---: | ---: | ---: | ---: |
| `ads_pi_cascade_ads_smoke` | baseline: `er=11.9`, `cond=5.8e7` | 1.9600e-2 | 4.8349e-2 | 1.71059 | 0.23045 |
| `ads_pi_cascade_ads_sweep_er108` | `er=10.8` | 2.5099e-2 | 4.2068e-2 | 2.16105 | 0.21557 |
| `ads_pi_cascade_ads_sweep_cond41e6` | `cond=4.1e7` | 1.9269e-2 | 4.8007e-2 | 1.68010 | 0.26365 |

On this small smoke set, lowering metal conductivity to `4.1e7 S/m` gives the
lowest test MSE, while lowering dielectric constant to `10.8` improves the
validation MSE but hurts test MSE. These are smoke-scale observations and should
be rechecked after increasing `MAX_SAMPLES`.

The baseline ADS smoke run also writes comparison figures to
`results/ads_pi_cascade_ads_smoke/comparison_plots/`.

## LHS200 Train Smoke

`ads_pi_cascade_lhs200_ads_smoke` uses `LHS200/train` for training and keeps
`LHS100/val` plus `LHS100/test` as validation/test splits because this checkout
does not contain `LHS200/val` or `LHS200/test`.

| Split | Count | Direct MSE Mean | Pi-NN MSE Mean | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 3 | 7.27940e-2 | 2.87827e-2 | 4.79777 | 0.23314 |
| train | 3 | 1.15764e-1 | 2.06084e-2 | 3.67652 | 0.31615 |
| val | 3 | 8.14023e-2 | 4.73266e-2 | 3.96921 | 0.31567 |

Comparison figures are in
`results/ads_pi_cascade_lhs200_ads_smoke/comparison_plots/`.

## LHS200 Random 100/100 Split

`train_ads_pi_cascade_v10.py` is configured to use only `LHS200/train` for the
default v10 ADS run. The script randomly shuffles the 200 available LHS200 rows
with seed `20260707`, selects 100 DUTs for modeling, and uses the remaining 100
DUTs for testing:

| Split | Source | DUT range | Count | Purpose |
| --- | --- | --- | ---: | --- |
| train | `HFSS_sim/LHS200/train/TSV_RDL` | random subset of `dut100`-`dut299` | 100 | Modeling |
| test | `HFSS_sim/LHS200/train/TSV_RDL` | remaining subset of `dut100`-`dut299` | 100 | Test |

Since there is no `LHS200/val` directory in this checkout, the modeling set is
also used as the validation mask for early stopping. Verification completed:

- `python -m py_compile model_versions/v10_ads_pi_cascade/code/train_ads_pi_cascade_v10.py model_versions/v10_ads_pi_cascade/rdl_ads_sim/ADS_Sim.py model_versions/v10_ads_pi_cascade/tsv_ads_sim/ADS_Sim.py`
- `collect_samples()` returned `{'train': 100, 'test': 100}` with zero overlap
  between the two random subsets.
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py`
  completed the full ADS simulation, pi optimization, parameter pretraining,
  and S-parameter fine-tuning flow.

The configured output directory is
`results/ads_pi_cascade_lhs200_random100train_100test/`.

| Split | Count | Direct MSE Mean | Pi-NN MSE Mean | Pi-NN MSE Median | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 100 | 9.9894e-2 | 1.7952e-2 | 1.1097e-2 | 3.21340 | 0.33362 |
| train | 100 | 1.09559e-1 | 1.0795e-2 | 5.367e-3 | 2.76629 | 0.22297 |

| Split | Direct NMSE Mean | Param-Pretrain NMSE Mean | Final Pi-NN NMSE Mean |
| --- | ---: | ---: | ---: |
| test | 0.238465 | 0.0417411 | 0.0383225 |
| train | 0.265234 | 0.0269300 | 0.0231390 |

Comparison figures are in
`results/ads_pi_cascade_lhs200_random100train_100test/comparison_plots/`.
They were regenerated with `code/regenerate_comparison_plots_v10.py` to compare
HFSS simulation, ADS direct cascade, optimized pi, and pi-NN model curves on
`S11 real`, `S11 imag`, `S21 real`, and `S21 imag`.
The current plot set contains 6 fixed-seed random test samples and 6 worst test
samples by `pi_nn_nmse_s11_s21_ri`.

The training entry now archives both model stages. The optimized connection
element dataset is `pi_optimized_targets.csv`; the preliminary model trained on
that dataset is `pi_connection_net_param_pretrain.pt`; the final model after
pure complex S-parameter fine-tuning is `pi_connection_net.pt`.

## Signed Pi Variant

`ads_pi_cascade_lhs200_random100train_100test_signed_pi` removes the
positive-only constraint on connection-network element scales. The optimizer
uses signed bounds `[-1e5, 1e5]`. Earlier signed-pi runs clipped the
neural-network output to the same signed range; current code leaves the
denormalized neural-network output unbounded during S-parameter training. ADS
single-device S-parameters are reused from the positive-pi cache; pi
optimization, model training, metrics, and plots are regenerated in the
signed-pi result directory.

| Split | Direct NMSE Mean | Optimized Pi NMSE Mean | Final Pi-NN NMSE Mean |
| --- | ---: | ---: | ---: |
| test | 0.238465 | 0.002705 | 0.057361 |
| train | 0.265234 | 0.003150 | 0.030930 |

The signed optimizer fits the training targets much more closely than the
positive-pi optimizer, but the final neural-network model is currently worse on
test NMSE than the positive-pi model, indicating the signed optimized element
values are harder for the present network/training setup to learn.

## Signed Pi 150/50 ADS Length 0.9

`ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09` is the current
default run. It uses the same LHS200 random seed, selects 150 samples for
modeling and 50 remaining samples for test, and scales ADS single-device
simulation lengths by `0.9` for `l_tmrdl`, `l_bsmrdl`, and `h_tsv`. The LHS
geometry columns used as neural-network features are not scaled.

| Split | Direct NMSE Mean | Optimized Pi NMSE Mean | Final Pi-NN NMSE Mean |
| --- | ---: | ---: | ---: |
| test | 0.465558 | 0.003346 | 0.049644 |
| train | 0.459582 | 0.003006 | 0.020003 |

The length scale makes the direct ADS cascade farther from the HFSS target than
the unscaled signed-pi run, but the optimized pi network still fits the target
closely. The final model improves over direct cascade but remains above the
per-sample optimized-pi lower bound.

## Unbounded S-Parameter Continuation

`ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_unbounded_sparam_continue`
continues from `ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09`
with no neural-network output range clamp after denormalization. The loss is
pure complex S-parameter loss with `PARAM_ANCHOR_WEIGHT = 0.0`; ADS
single-device S-parameters are loaded from the source run cache.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\continue_sparam_unbounded_v10.py model_versions\v10_ads_pi_cascade\code\regenerate_comparison_plots_v10.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\continue_sparam_unbounded_v10.py`
- Generated 12 comparison figures under `comparison_plots/`.

Before continuation:

| Split | Direct NMSE Mean | Final Pi-NN NMSE Mean | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: |
| test | 0.465558 | 0.049644 | 3.107216 | 0.472894 |
| train | 0.459582 | 0.020003 | 2.591103 | 0.319014 |

After unbounded S-parameter continuation:

| Split | Direct NMSE Mean | Final Pi-NN NMSE Mean | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: |
| test | 0.465558 | 0.043168 | 2.886461 | 0.423930 |
| train | 0.459582 | 0.011813 | 2.101195 | 0.229016 |

The unbounded S-parameter continuation improves the held-out 50-sample test
NMSE from `0.049644` to `0.043168`.

## V09-Style Network Trial

`ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_v09net`
reuses the ADS cache and optimized pi target dataset from the current default
150/50 run, but replaces the small v10 network with the v09-style larger
multi-head network:

- Shared trunk: `9 -> 256 -> 256 -> 128` with `SiLU` and `LayerNorm`.
- Eight heads: each `128 -> 64 -> 4`, one head per pi insertion position.
- Output: 32 v10 pi parameters, with no denormalized output clamp.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_v09net.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_v09net.py`
- Generated 12 comparison figures under `comparison_plots/`.

| Model | Split | Final Pi-NN NMSE Mean | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | --- | ---: | ---: | ---: |
| small v10 + unbounded continue | test | 0.043168 | 2.886461 | 0.423930 |
| v09-style net | test | 0.059483 | 3.295351 | 0.521695 |
| small v10 + unbounded continue | train | 0.011813 | 2.101195 | 0.229016 |
| v09-style net | train | 0.002815 | 0.557305 | 0.072331 |

The larger v09-style network strongly improves the modeling-set fit but hurts
the held-out 50-sample test result, so this trial indicates overfitting rather
than better generalization.

## LHS800 Training Trial

`ads_pi_cascade_lhs800train_lhs200test_signed_pi_adslen09` uses all 800 samples
from `HFSS_sim/LHS800/train/TSV_RDL` for training. The fixed 50-sample LHS200
holdout from the current 150/50 run is used as an external test set. The model
uses the current small v10 `PiConnectionNet`, ADS single-device length scale
`0.9`, signed pi optimization bounds, and no denormalized NN output clamp.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_lhs800.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_lhs800.py`
- Generated 12 comparison figures under `comparison_plots/`.

| Split | Count | Direct NMSE Mean | Final Pi-NN NMSE Mean | Final Pi-NN NMSE Median | Pi-NN S11 MAE dB | Pi-NN S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.465558 | 0.091791 | 0.019139 | 3.155421 | 0.656975 |
| train | 800 | 0.434526 | 0.004912 | 0.003147 | 1.230639 | 0.137637 |

Compared with the 150-sample unbounded continuation run, LHS800 training makes
the training fit much better but worsens held-out test mean NMSE
(`0.043168 -> 0.091791`). The test median improves (`0.032776 -> 0.019139`),
so the mean degradation is driven by a few bad LHS200 holdout samples.

## Calibrated 16-DUT ADS Settings, LHS200 Random 100/100

`ads_pi_cascade_lhs200_random100train_100test_calibrated16dut` uses the
16-DUT single-device calibrated ADS settings:

- RDL: `er_si=10.2`, `cond=5.8e7`, `tand=0.005`, `l_scale=1.0`,
  `w_scale=0.85`, `pitch_scale=1.0`, `h_tsv_scale=1.0`, `h_rdl_scale=0.8`.
- TSV: `er_si=11.9`, `cond=5.8e7`, `tand=0.005`, `c1_scale=1.0`,
  `pitch_scale=1.0`, `h_tsv_scale=1.1`, `d_scale=1.0`.

The training flow is unchanged: ADS single-device simulation, eight pi-network
optimization sets per sample, structure-to-pi parameter pretraining, then pure
complex S-parameter fine-tuning. The random split uses seed `20260707` with 100
LHS200 samples for modeling and the remaining 100 for test. This run uses
`ADS_DEVICE_LENGTH_SCALE=1.0` so that the geometry changes come from the
calibrated ADS settings rather than an extra global length multiplier.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_calibrated16dut_lhs200_random100.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- Split check: `train=100`, `test=100`, overlap `0`.
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_calibrated16dut_lhs200_random100.py`
- Generated 12 comparison figures under `comparison_plots/`.

After parameter pretraining:

| Split | Count | Direct NMSE Mean | Param-Pretrain NMSE Mean | Param-Pretrain NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 100 | 0.254311 | 0.073972 | 0.055380 | 3.407290 | 0.506764 |
| train | 100 | 0.293169 | 0.057745 | 0.035060 | 3.237287 | 0.366871 |

After final S-parameter fine-tuning:

| Split | Count | Direct NMSE Mean | Final Pi-NN NMSE Mean | Final Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 100 | 0.254311 | 0.046348 | 0.031626 | 2.952143 | 0.406668 |
| train | 100 | 0.293169 | 0.030046 | 0.016669 | 2.651979 | 0.276429 |

Compared with the earlier random 100/100 v10 result (`test direct NMSE
0.238465`, `test final Pi-NN NMSE 0.038322`), the calibrated ADS settings do
not improve the held-out test metric in this full cascade training run.

Additional best-test comparison plots were generated with
`code/plot_best_calibrated16dut_lhs200_random100.py`:

- `comparison_plots/best_test/`: 8 plots selected by lowest test
  `pi_nn_nmse_s11_s21_ri`.
- `best_test_comparison_plots.csv`: selected sample metrics and plot paths.
- `best_test_comparison_plots_validation.md`: validation archive for this plot
  run.

Best plotted test samples:

| Sample | DUT | Pi-NN NMSE | Direct NMSE | Optimized Pi NMSE |
| --- | ---: | ---: | ---: | ---: |
| LHS200_train_dut240 | 240 | 0.003328 | 0.061948 | 9.734e-07 |
| LHS200_train_dut116 | 116 | 0.003965 | 0.091488 | 7.653e-07 |
| LHS200_train_dut188 | 188 | 0.006238 | 0.166973 | 2.481e-05 |
| LHS200_train_dut158 | 158 | 0.007986 | 0.096694 | 4.298e-05 |
| LHS200_train_dut121 | 121 | 0.008420 | 0.295343 | 7.849e-04 |
| LHS200_train_dut105 | 105 | 0.009043 | 0.422448 | 2.512e-05 |
| LHS200_train_dut142 | 142 | 0.009548 | 0.141432 | 1.042e-04 |
| LHS200_train_dut122 | 122 | 0.009716 | 0.219676 | 2.671e-03 |

## LHS150_50_Connection2, 0.1-100 GHz

`ads_pi_cascade_lhs150_50_connection2_100ghz_calibrated16dut` uses
`HFSS_sim/LHS150_50_Connection2/train` for modeling and
`HFSS_sim/LHS150_50_Connection2/test` for testing. The frequency grid is the
native HFSS grid, 1000 points from 0.1 to 100 GHz. The modeling flow is
unchanged: ADS single-device simulation, eight pi-network optimization sets per
sample, structure-to-pi parameter pretraining, then pure complex S-parameter
fine-tuning.

Implementation notes:

- `rdl_ads_sim/ADS_Sim.py` and `tsv_ads_sim/ADS_Sim.py` now accept
  `freq_start_ghz`, `freq_stop_ghz`, and `freq_step_ghz` in `ads_settings`.
- This run sets ADS sweep to `Start=0.1 GHz`, `Stop=100 GHz`, `Step=0.1 GHz`.
- `TSV_RDL_variations_record.csv` columns `t_tmrdl` and `t_bsmrdl` are mapped
  to the v10 model inputs `h_tmrdl` and `h_bsmrdl`.
- The calibrated 16-DUT ADS settings are reused, with `ADS_DEVICE_LENGTH_SCALE=1.0`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_150_50_100ghz.py`
- Sample check: `train=150`, `test=50`, frequency grid `1000` points from
  `0.1` to `100.0` GHz.
- ADS generated netlists checked with `SweepPlan: SP1_stim Start=0.1 GHz Stop=100 GHz Step=0.1 GHz`.
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_150_50_100ghz.py`
- Generated 12 comparison figures under `comparison_plots/`.

Per-sample optimized pi result:

| Split | Count | Direct NMSE Mean | Optimized Pi NMSE Mean | Optimized Pi MSE Mean |
| --- | ---: | ---: | ---: | ---: |
| test | 50 | 0.922865 | 0.247965 | 0.120506 |
| train | 150 | 0.921287 | 0.270185 | 0.127627 |

After parameter pretraining:

| Split | Count | Direct NMSE Mean | Param-Pretrain NMSE Mean | Param-Pretrain NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.922865 | 1.084472 | 0.976982 | 6.519040 | 26.027263 |
| train | 150 | 0.921287 | 1.001825 | 0.935731 | 6.247329 | 24.417257 |

After final S-parameter fine-tuning:

| Split | Count | Direct NMSE Mean | Final Pi-NN NMSE Mean | Final Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.922865 | 0.823552 | 0.811913 | 5.965828 | 23.157745 |
| train | 150 | 0.921287 | 0.764886 | 0.755957 | 5.752200 | 22.471483 |

The 0.1-100 GHz run shows that per-sample optimized pi can reduce test NMSE
from about `0.923` to `0.248`, but the current small Pi-NN does not learn that
mapping well over the wider frequency range.

## LHS400_Connection2 Random-10 RDL Calibration, MLIN

`ads_cal_rdl_lhs400c2_rand10_mlin` recalibrates the updated RDL ADS `MLIN2`
template on 10 random `HFSS_sim/LHS400_Connection2/train/RDL` samples. The
random seed is `20260708`; selected DUTs are `12, 45, 116, 150, 187, 242, 269,
298, 335, 373`. The ADS sweep is `0.1-100 GHz` with `0.1 GHz` step.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_rdl_lhs400_connection2_random10.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_rdl_lhs400_connection2_random10.py`
- Generated 5 comparison figures under `plots/rdl/`.

Best RDL settings:

| er_si | cond | tand | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.8 | 5.8e7 | 0.005 | 1.0 | 0.8 | 1.1 | 1.0 | 1.0 |

Top result:

| Candidate | Count | NMSE Mean | NMSE Median | NMSE Max | MSE Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| er9.8_w0.8_pitch1.1 | 10 | 0.082460 | 0.078613 | 0.114453 | 0.038907 |

## LHS400_Connection2 Random-10 RDL and TSV Calibration

`ac_l400_rdl_tsv10` calibrates the current ADS RDL `MLIN2` template and the
current ADS TSV `d_tsv` template on random single-device samples from
`HFSS_sim/LHS400_Connection2/train`. The ADS sweep is `0.1-100 GHz` with
`0.1 GHz` step.

Sample selection:

- RDL seed `20260708`, DUTs `12, 45, 116, 150, 187, 242, 269, 298, 335, 373`.
- TSV seed `20260709`, DUTs `20, 85, 95, 143, 191, 217, 267, 316, 330, 338`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\calibrate_ads_lhs400_connection2_rdl_tsv_random10.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_lhs400_connection2_rdl_tsv_random10.py`
- Generated 10 comparison figures under `plots/rdl/` and `plots/tsv/`.
- Generated netlists checked: RDL uses `MLIN2`, TSV uses `d_tsv`, and both use
  `SweepPlan: SP1_stim Start=0.1 GHz Stop=100 GHz Step=0.1 GHz`.

Best RDL settings:

| er_si | cond | tand | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | NMSE Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.4 | 5.8e7 | 0.005 | 1.0 | 0.75 | 1.15 | 1.0 | 1.0 | 0.062834 |

Best TSV settings:

| er_si | cond | tand | c1_scale | pitch_scale | h_tsv_scale | d_scale | NMSE Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11.9 | 5.8e7 | 0.005 | 1.0 | 0.9 | 1.1 | 1.0 | 0.009993 |

## LHS400_Connection2 Random-10 RDL and TSV Refined Calibration

`ac_l400_ref2` refines the previous random-10 calibration around
`ac_l400_rdl_tsv10` best settings. It uses the same sampled DUTs and the same
`0.1-100 GHz` ADS sweep.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\calibrate_ads_lhs400_connection2_rdl_tsv_random10_refined.py model_versions\v10_ads_pi_cascade\code\calibrate_ads_lhs400_connection2_rdl_tsv_random10.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\calibrate_ads_lhs400_connection2_rdl_tsv_random10_refined.py`
- Generated 10 comparison figures under `plots/rdl/` and `plots/tsv/`.

Best refined RDL settings:

| er_si | cond | tand | l_scale | w_scale | pitch_scale | h_tsv_scale | h_rdl_scale | NMSE Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.8 | 5.8e7 | 0.005 | 1.0 | 0.65 | 1.25 | 1.0 | 1.0 | 0.039423 |

Best refined TSV settings:

| er_si | cond | tand | c1_scale | pitch_scale | h_tsv_scale | d_scale | NMSE Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11.9 | 5.8e7 | 0.005 | 1.0 | 1.0 | 1.2 | 1.0 | 0.007154 |

## LHS150_50_Connection2 Cascade With Refined LHS400 ADS

`ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads` reruns the
standard v10 ADS pi-cascade modeling flow on `LHS150_50_Connection2`, using
the refined single-device ADS settings from `ac_l400_ref2`. The modeling
process is unchanged: ADS single-device simulation, per-sample eight-pi
optimization, structure-to-pi parameter pretraining, then S-parameter
fine-tuning.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\rdl_ads_sim\ADS_Sim.py model_versions\v10_ads_pi_cascade\tsv_ads_sim\ADS_Sim.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py`
- Generated 600 single-device ADS `.s2p` cache files and 12 comparison plots.

Per-sample optimized pi result:

| Split | Count | Direct NMSE Mean | Optimized Pi NMSE Mean | Optimized Pi MSE Mean |
| --- | ---: | ---: | ---: | ---: |
| test | 50 | 1.889792 | 0.020980 | 0.009078 |
| train | 150 | 1.861976 | 0.012171 | 0.005175 |

After parameter pretraining:

| Split | Count | Direct NMSE Mean | Param-Pretrain NMSE Mean | Param-Pretrain NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 1.889792 | 0.645118 | 0.575530 | 4.937626 | 6.546289 |
| train | 150 | 1.861976 | 0.535539 | 0.389284 | 4.490136 | 5.055235 |

After final S-parameter fine-tuning:

| Split | Count | Direct NMSE Mean | Final Pi-NN NMSE Mean | Final Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 1.889792 | 0.549241 | 0.466522 | 4.878773 | 5.348503 |
| train | 150 | 1.861976 | 0.362609 | 0.189431 | 4.199485 | 3.772976 |

## Refined LHS400 ADS S-Parameter Continuation

`ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_sparam_continue`
continues from the refined-LHS400 ADS Connection2 checkpoint and trains only
with pure complex S-parameter loss. It reuses the existing ADS single-device
cache and optimized pi targets; the modeling data split is unchanged.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\continue_sparam_connection2_refined_lhs400_ads.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\continue_sparam_connection2_refined_lhs400_ads.py`
- Continue epochs: `160`; learning rate: `8e-6`; patience: `40`.
- Generated 12 comparison plots under `comparison_plots/`.

Before continuation:

| Split | Count | Final Pi-NN NMSE Mean | Final Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.549241 | 0.466522 | 4.878773 | 5.348503 |
| train | 150 | 0.362609 | 0.189431 | 4.199485 | 3.772976 |

After S-parameter continuation:

| Split | Count | Continued Pi-NN NMSE Mean | Continued Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.476691 | 0.324145 | 4.810463 | 4.261821 |
| train | 150 | 0.229312 | 0.102102 | 3.897470 | 2.694704 |

## Filtered Training Trial

`ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_filtered_train20`
reuses the refined-LHS400 ADS cache and optimized pi targets, excludes the 20
highest-error training samples from the S-parameter continuation metrics, and
restarts the neural-network training with the same parameter-pretrain then
S-parameter-training flow. The 50 test samples are not filtered.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_filtered_connection2_refined_lhs400_ads.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_filtered_connection2_refined_lhs400_ads.py`
- Excluded samples are recorded in `excluded_train_samples.csv`.
- Generated 12 comparison plots under `comparison_plots/`.

Final filtered-train result:

| Split | Count | Pi-NN NMSE Mean | Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.520745 | 0.405490 | 4.920713 | 5.407574 |
| train | 130 | 0.178825 | 0.094406 | 3.722011 | 1.543948 |
| excluded_train | 20 | 0.912100 | 0.926449 | 5.026988 | 11.827685 |

This filtered retraining improves over the pre-continuation full-data model
(`test NMSE mean 0.549241`) but is worse than the S-parameter continuation
model (`test NMSE mean 0.476691`).

## Element-Wise Multi-Head Architecture Trial

`ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_element_heads`
uses the requested element-wise neural-network structure. For each element
type, the network has a shared `9 -> 30 -> 30` trunk and eight independent
connection-position heads of `30 -> 20 -> 1`. The output remains 32 pi-network
parameters.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10.py model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_refined_lhs400_ads_element_heads.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_pi_cascade_v10_connection2_refined_lhs400_ads_element_heads.py`
- Generated 12 comparison plots under `comparison_plots/`.

Final result:

| Split | Count | Pi-NN NMSE Mean | Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 50 | 0.460406 | 0.415037 | 4.968888 | 3.660034 |
| train | 150 | 0.428516 | 0.353032 | 4.609605 | 3.130634 |

This is currently the best test NMSE among the tried neural-network variants:
it improves over S-parameter continuation (`0.476691`) and filtered retraining
(`0.520745`), but remains far above per-sample optimized pi (`0.020980`).

## Optimized-Pi Filtered Training Trial

`ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_optfilter005`
filters samples before neural-network training using per-sample optimized-pi
error. Samples with `optimized_pi_nmse_s11_s21_ri > 0.05` are excluded from the
active train/test sets and reported separately.

Excluded samples:

- train: 2 samples, DUTs `76`, `140`.
- test: 3 samples, DUTs `171`, `185`, `199`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_optfilter_connection2_refined_lhs400_ads.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_optfilter_connection2_refined_lhs400_ads.py`
- Excluded samples are recorded in `excluded_optimized_pi_samples.csv`.
- Generated 12 comparison plots under `comparison_plots/`.

Final result:

| Split | Count | Pi-NN NMSE Mean | Pi-NN NMSE Median | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 47 | 0.444260 | 0.398114 | 4.953595 | 3.761811 |
| train | 148 | 0.395850 | 0.267383 | 4.504485 | 3.208809 |
| excluded_opt_test | 3 | 0.799328 | 0.711378 | 5.318232 | 10.638370 |
| excluded_opt_train | 2 | 0.528596 | 0.528596 | 4.020843 | 6.592763 |

The kept-test NMSE is lower than the full 50-sample element-wise run
(`0.460406`), but the comparison is not one-to-one because three hard-to-fit
test samples were removed from the active test set.

## V08-Circuit Shared-to-Multihead Trial

`ads_v08circuit_shared_to_multihead_lhs150_50_connection2_100ghz_lhs400_refined_ads`
uses the refined-LHS400 ADS single-device cache and replaces the v10 four-element
pi network with the v08 with-Cn3 connection circuit. In the optimization stage,
each full-cascade sample fits one shared 7-parameter circuit and inserts that
same circuit at all eight connection positions. The first neural network stage
trains seven independent `9->30->30->20->1` scalar networks against those shared
parameters. The S-parameter stage initializes from that model, expands each
parameter network into eight `30->20->1` connection-position heads, and trains
against the `S11`/`S21` magnitude and wrapped phase target.
Before neural-network training, the script now reviews the optimized shared-
circuit NMSE and excludes samples with
`optimized_v08_shared_nmse_s11_s21_ri > 0.3`.

Validation completed:

- `conda run -n PyML python -m py_compile model_versions\v10_ads_pi_cascade\code\train_ads_v08circuit_shared_to_multihead_connection2_refined_lhs400_ads.py`
- `conda run -n PyML python model_versions\v10_ads_pi_cascade\code\train_ads_v08circuit_shared_to_multihead_connection2_refined_lhs400_ads.py`
- Excluded optimized samples are recorded in `excluded_optimized_v08_shared_samples.csv`.
- Generated comparison plots under `comparison_plots/`.
- Validation archive: `model_versions/v10_ads_pi_cascade/results/ads_v08circuit_shared_to_multihead_lhs150_50_connection2_100ghz_lhs400_refined_ads/validation_archive.md`

Per-sample shared-circuit optimization:

| Split | Count | Direct NMSE Mean | Optimized Shared V08 NMSE Mean | Optimized Shared V08 NMSE Median |
| --- | ---: | ---: | ---: | ---: |
| test | 50 | 1.889792 | 0.176276 | 0.135372 |
| train | 150 | 1.861976 | 0.180935 | 0.152259 |

Optimized-result filter:

| Original Split | Excluded Count | Excluded Optimized NMSE Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| test | 11 | 0.395025 | 0.306843 | 0.580906 |
| train | 33 | 0.396537 | 0.303733 | 0.576336 |

Active train/test after filtering: `117 / 39`.

After shared-parameter pretraining:

| Split | Count | V08-NN NMSE Mean | V08-NN NMSE Median | V08-NN Mag/Phase MSE Mean | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| excluded_opt_test | 11 | 0.584779 | 0.644324 | 1.287430 | 5.755080 | 10.939900 |
| excluded_opt_train | 33 | 0.618043 | 0.639606 | 1.168550 | 5.381420 | 10.893800 |
| test | 39 | 0.448807 | 0.319553 | 0.842558 | 5.616950 | 4.316650 |
| train | 117 | 0.168035 | 0.139176 | 0.556563 | 5.422120 | 2.059910 |

After S11/S21 magnitude-phase fine-tuning:

| Split | Count | V08-NN NMSE Mean | V08-NN NMSE Median | V08-NN Mag/Phase MSE Mean | S11 MAE dB | S21 MAE dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| excluded_opt_test | 11 | 0.574713 | 0.581659 | 1.151840 | 5.487553 | 8.227207 |
| excluded_opt_train | 33 | 0.592529 | 0.601021 | 1.084379 | 4.986087 | 10.037617 |
| test | 39 | 0.389323 | 0.300856 | 0.680871 | 5.334023 | 3.570800 |
| train | 117 | 0.132654 | 0.108897 | 0.304275 | 4.737773 | 2.054155 |

After filtering poor optimized samples, the active-test NMSE mean is `0.389323`
on 39 samples, compared with `0.433238` on the full 50-sample test set before
filtering. This comparison is not one-to-one because 11 test samples are reported
separately under `excluded_opt_test`.
