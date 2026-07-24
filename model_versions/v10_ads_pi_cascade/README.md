# v10 ADS Pi Cascade

This version keeps the v09 training sequence but changes the cascade correction
network to eight inserted pi networks. RDL and TSV single-device S-parameters
come from ADS through separate Python runners in `rdl_ads_sim/ADS_Sim.py` and
`tsv_ads_sim/ADS_Sim.py`, which are called by
`code/train_ads_pi_cascade_v10.py` when `SIMULATION_BACKEND = "ads"`.

The workflow is:

1. Simulate single `TMRDL`, `TSV`, and `BSMRDL` S-parameters, cascade the
   device blocks, then optimize eight pi-network element sets against the
   complete `TSV_RDL` target S-parameters. The optimized element-value dataset
   is saved as `pi_optimized_targets.csv`.
2. Train a neural network from structure parameters to the optimized pi-network
   element values, and save this preliminary model as
   `pi_connection_net_param_pretrain.pt`.
3. Fine-tune the pi-network neural network again with `S11`/`S21` magnitude
   and wrapped phase loss. The final S-parameter-tuned model is saved as
   `pi_connection_net.pt`.

Model error is reported with NMSE:
`sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)`, where `y` is the
flattened vector of `S11.real`, `S11.imag`, `S21.real`, and `S21.imag` over all
frequency points. The summary CSV keeps the historical complex MSE columns and
adds `*_nmse_s11_s21_ri_*` columns for this formula.

`code/train_ads_pi_cascade_v10.py` is a direct-run VS Code entry point. Its
top-level constants configure the ADS backend switch, material settings, source
dataset split, and training epochs. The current default uses `LHS200/train`
only: 150 DUTs are randomly selected for modeling with seed `20260707`, and the
remaining 50 DUTs are used for testing. Because `LHS200` has no independent
validation split in this checkout, the modeling set is reused for early-stop
validation during training.

The current run label is the signed-pi variant. Connection-network element
scales are allowed to be negative with optimizer bounds `[-1e5, 1e5]`;
neural-network outputs are now unbounded after denormalization, so S-parameter
continuation can move outside the optimizer target range if the S-parameter
loss improves.
For ADS single-device simulation, the RDL lengths `l_tmrdl`/`l_bsmrdl` and TSV
length `h_tsv` are scaled by `0.9`; the neural-network input features keep the
original LHS geometry values.

Current `PiConnectionNet` architecture is element-wise. For each pi element
type (`Cleft_scale`, `Rseries_scale`, `Lseries_scale`, `Cright_scale`) there is
a shared `9 -> 30 -> 30` trunk, followed by eight connection-position heads.
Each position head is `30 -> 20 -> 1`, so the four element networks produce
the same 32 outputs as before.

## Code

| Script | Purpose |
| --- | --- |
| `code/calibrate_ads_single_devices_v10.py` | Uses a small HFSS single-device subset to calibrate ADS RDL/TSV settings. Primary sweep variables are `er_si`, `cond`, `tand`, and TSV `c1_scale`; secondary variables are geometry scale factors. |
| `code/calibrate_ads_single_devices_v10_refined.py` | Refines the ADS single-device calibration with a larger six-DUT subset and combined candidate settings around the first calibration result. |
| `code/calibrate_ads_single_devices_v10_16dut.py` | Runs the same refined ADS calibration search on 16 evenly spaced LHS200 DUTs for a broader calibration set. |
| `code/calibrate_ads_single_devices_v10_16dut_rdl_net_update.py` | Recalibrates RDL ADS settings on the 16-DUT subset after the RDL ADS netlist update, writing to a fresh short-path result directory. |
| `code/calibrate_ads_rdl_lhs400_connection2_random10.py` | Uses the updated RDL ADS MLIN template and calibrates ADS RDL settings on 10 random `LHS400_Connection2/train/RDL` samples over 0.1-100 GHz. |
| `code/calibrate_ads_lhs400_connection2_rdl_tsv_random10.py` | Calibrates ADS RDL and TSV settings on 10 random `LHS400_Connection2/train/RDL` samples and 10 random `LHS400_Connection2/train/TSV` samples over 0.1-100 GHz. |
| `code/calibrate_ads_lhs400_connection2_rdl_tsv_random10_refined.py` | Refines the LHS400_Connection2 random-10 ADS RDL/TSV calibration around the previous best settings. |
| `code/plot_ads_single_device_calibration_16dut.py` | Regenerates additional best-setting comparison plots for every 16-DUT calibration sample without rerunning the calibration sweep. |
| `code/plot_best_calibrated16dut_lhs200_random100.py` | Reloads the calibrated 16-DUT LHS200 100/100 checkpoint and saves best-test comparison plots by final Pi-NN NMSE. |
| `code/train_ads_pi_cascade_v10.py` | Runs ADS or cached single-device simulation, pi-network optimization, pi-parameter NN training, and final S-parameter fine-tuning. |
| `code/train_ads_pi_cascade_v10_calibrated16dut_lhs200_random100.py` | Runs the standard v10 pi-cascade training flow on a random LHS200 100/100 split using the 16-DUT calibrated RDL/TSV ADS settings. |
| `code/train_ads_pi_cascade_v10_connection2_150_50_100ghz.py` | Runs the standard v10 pi-cascade training flow on `HFSS_sim/LHS150_50_Connection2`, using its 150 train and 50 test samples over 0.1-100 GHz. |
| `code/train_ads_pi_cascade_v10_connection2_refined_lhs400_ads.py` | Runs the standard v10 pi-cascade training flow on `LHS150_50_Connection2` using the refined `ac_l400_ref2` ADS RDL/TSV settings. |
| `code/train_ads_pi_cascade_v10_connection2_refined_lhs400_ads_element_heads.py` | Runs the refined-LHS400 ADS Connection2 flow with the element-wise `9->30->30` shared trunks and `30->20->1` connection heads. |
| `code/train_ads_v08circuit_shared_to_multihead_connection2_refined_lhs400_ads.py` | Runs the refined-LHS400 ADS Connection2 flow with the v08 with-Cn3 connection circuit. Optimization fits one shared 7-parameter circuit per sample, pretrains seven `9->30->30->20->1` scalar networks, then expands each parameter network into eight connection-position heads for S-parameter fine-tuning. |
| `code/train_ads_pi_cascade_v10_lhs800.py` | Trains the current small v10 Pi-NN with all 800 LHS800 samples and evaluates on the fixed 50-sample LHS200 holdout. |
| `code/train_ads_pi_cascade_v10_v09net.py` | Reuses the current 150/50 ADS cache and optimized pi targets, then trains a v09-style larger multi-head network adapted to the v10 four-parameter pi heads. |
| `code/continue_sparam_unbounded_v10.py` | Reloads the current 150/50 signed-pi checkpoint, removes the output range clamp, and continues training against the `S11`/`S21` magnitude and wrapped phase target. |
| `code/continue_sparam_connection2_refined_lhs400_ads.py` | Reloads the refined-LHS400 ADS Connection2 checkpoint and continues training against the `S11`/`S21` magnitude and wrapped phase target. |
| `code/train_filtered_connection2_refined_lhs400_ads.py` | Retrains the refined-LHS400 ADS Connection2 neural network after excluding the highest-error training samples from the S-parameter continuation run. |
| `code/train_optfilter_connection2_refined_lhs400_ads.py` | Retrains after excluding train and test samples whose per-sample optimized-pi NMSE is above the configured threshold. |
| `code/regenerate_comparison_plots_v10.py` | Reloads the saved checkpoint and cached ADS single-device data, then regenerates comparison plots without rerunning training. |
| `rdl_ads_sim/ADS_Sim.py` | ADS RDL helper for `TMRDL` and `BSMRDL`; maps RDL geometry/material parameters into the RDL `sim.net`, writes generated netlists, calls `hpeesofsim.exe`, and returns `.s2p`. |
| `tsv_ads_sim/ADS_Sim.py` | ADS TSV helper; maps TSV radius/height/pitch and material parameters into the TSV `sim.net`, writes generated netlists, calls `hpeesofsim.exe`, and returns `.s2p`. |

## Results

| Path | Content |
| --- | --- |
| `results/ads_single_device_calibration_small/` | Small-sample ADS-vs-HFSS single-device calibration for RDL and TSV. Contains detail/summary CSVs, best settings JSON, validation archive, ADS cache, and comparison plots. |
| `results/ads_single_device_calibration_refined/` | Refined ADS-vs-HFSS single-device calibration using six LHS200 DUTs and combined RDL/TSV candidate settings. |
| `results/ads_single_device_calibration_16dut/` | Broader ADS-vs-HFSS single-device calibration using 16 evenly spaced LHS200 DUTs and the refined candidate search. `plots_all_best/` contains 48 best-setting comparison plots covering TMRDL, BSMRDL, and TSV for all 16 DUTs. |
| `results/ads_cal_rdl_update16/` | RDL-only 16-DUT recalibration after the RDL ADS netlist update. TSV settings are carried over from `ads_single_device_calibration_16dut`. |
| `results/ads_cal_rdl_lhs400c2_rand10/` | Original-MCLIN ADS RDL calibration on 10 random `LHS400_Connection2/train/RDL` samples over 0.1-100 GHz. |
| `results/ads_cal_rdl_lhs400c2_rand10_mlin/` | Updated-MLIN ADS RDL calibration on the same 10 random `LHS400_Connection2/train/RDL` samples over 0.1-100 GHz. |
| `results/ac_l400_rdl_tsv10/` | Updated-MLIN RDL and d_tsv TSV ADS calibration on 10 random LHS400_Connection2 RDL samples and 10 random TSV samples. |
| `results/ac_l400_ref2/` | Refined updated-MLIN RDL and d_tsv TSV ADS calibration around `ac_l400_rdl_tsv10` best settings. |
| `results/ads_pi_cascade_lhs200_random100train_100test/` | Default LHS200 random 100-modeling/100-test ADS outputs, metrics, model checkpoint, and S11/S21 real-imag plots comparing HFSS simulation, ADS direct cascade, optimized pi, and pi-NN model results. Comparison plots are split into fixed-seed random test samples and worst test samples by NMSE. |
| `results/ads_pi_cascade_lhs200_random100train_100test_calibrated16dut/` | LHS200 random 100/100 v10 pi-cascade run using the 16-DUT calibrated ADS settings and no extra global length multiplier. `comparison_plots/best_test/` contains the best test-sample plots by final Pi-NN NMSE. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_calibrated16dut/` | `LHS150_50_Connection2` v10 pi-cascade run using 150 train and 50 test samples over 0.1-100 GHz with calibrated ADS settings. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads/` | `LHS150_50_Connection2` v10 pi-cascade run using refined LHS400 ADS RDL/TSV settings from `ac_l400_ref2`. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_sparam_continue/` | Pure S-parameter continuation from the refined-LHS400 ADS Connection2 checkpoint. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_filtered_train20/` | Filtered retraining trial excluding the 20 highest-error training samples from the S-parameter continuation run. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_element_heads/` | Refined-LHS400 ADS Connection2 run using the element-wise shared-trunk multi-head architecture. |
| `results/ads_pi_cascade_lhs150_50_connection2_100ghz_lhs400_refined_ads_optfilter005/` | Element-wise network retraining after excluding samples with optimized-pi NMSE greater than `0.05`. |
| `results/ads_v08circuit_shared_to_multihead_lhs150_50_connection2_100ghz_lhs400_refined_ads/` | V08 with-Cn3 connection-circuit trial. Per-sample optimization uses one shared 7-parameter circuit for all eight connection positions; samples with optimized shared-circuit NMSE above `0.3` are excluded before NN training. Active train/test is `117/39`; final active-test NMSE mean is `0.389323`; excluded samples are listed in `excluded_optimized_v08_shared_samples.csv`. |
| `results/ads_pi_cascade_lhs200_random100train_100test_signed_pi/` | Signed-pi run with the same random 100/100 split and no positive-only constraint on connection-network element scales. This run reuses the ADS single-device cache from the positive-pi run but recomputes pi optimization, training, metrics, and plots. |
| `results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09/` | Current default signed-pi run with 150 random LHS200 samples for modeling, 50 remaining samples for test, and ADS RDL/TSV lengths scaled by `0.9`. |
| `results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_unbounded_sparam_continue/` | Continuation from the current default checkpoint with no neural-network output range clamp and pure S-parameter loss. |
| `results/ads_pi_cascade_lhs200_random150train_50test_signed_pi_adslen09_v09net/` | v09-style larger multi-head NN trial. It fits the 150-sample modeling set much better but is worse on the 50-sample test set. |
| `results/ads_pi_cascade_lhs800train_lhs200test_signed_pi_adslen09/` | LHS800 trial using all 800 LHS800 samples for training and the fixed 50-sample LHS200 holdout for testing. |
