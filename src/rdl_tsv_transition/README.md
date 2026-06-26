# rdl_tsv_transition

RDL/TSV 绾ц仈銆佽繃娓＄粨鏋勫缓妯°€佸叡浜繃娓＄粨鏋勭缁忕綉缁滆缁冨拰 HFSS 绔埌绔井璋冨伐鍏峰寘銆?
鍘熷鍗曟枃浠惰剼鏈凡鎷嗗垎涓哄涓ā鍧椼€傚吋瀹瑰叆鍙ｄ粛淇濈暀鍦ㄤ笂涓€绾х洰褰曪細

```text
../rdl_tsv_transition_dataset_train.py
```

鎺ㄨ崘鐩存帴浠庤鍏ュ彛杩愯锛屾垨浠庢湰鍖呭鍏?`run_dataset_training`銆?
## 渚濊禆

杩愯瀹屾暣娴佺▼闇€瑕侊細

```text
numpy
scipy
scikit-rf
matplotlib
torch
```

濡傛灉瀵煎叆鏃舵姤閿?`ModuleNotFoundError: No module named 'skrf'`锛岄渶瑕佸厛瀹夎 `scikit-rf`銆?
## 鏁版嵁鐩綍绾﹀畾

榛樿鍋囪鏁版嵁鍜?MATLAB 瀵煎嚭鐨勭綉缁滃弬鏁颁綅浜庡叆鍙ｈ剼鏈悓绾х洰褰曪細

```text
snp_data/RDL_TSV_Snp/
  dut1.s2p
  dut2.s2p
  ...

device_models/RDL_TSV_mat2/
  RDL_Top_R1.mat
  RDL_Top_R2.mat
  ...
  RDL_Bottom_R1.mat
  ...
  TSV_R1.mat
  ...
```

`.s2p` 鏂囦欢澶撮儴娉ㄩ噴涓渶瑕佸寘鍚嚑浣曞弬鏁帮紝渚嬪锛?
```text
! lrdl=...
! wrdl=...
! trdl=...
! ldown=...
! wdown=...
! tdown=...
! dtsv=...
! htsv=...
! p1=...
# ...
```

## 妯″潡璇存槑

### `constants.py`

瀹氫箟鍏ㄥ眬甯搁噺鍜屾ā鍨嬬害瀹氾細

- `Z_REF`锛氬弬鑰冮樆鎶楋紝榛樿 50 Ohm銆?- `CIRCUIT_PARAM_NAMES`锛歁ATLAB 缃戠粶杈撳嚭鐨勭瓑鏁堢數璺弬鏁板悕銆?- `DEVICE_SEQUENCE`锛氭暣浣撶粨鏋勪腑鐨?RDL/TSV 绾ц仈椤哄簭銆?- `MAT_PREFIX`锛氫笉鍚屽櫒浠剁被鍨嬪搴旂殑 `.mat` 鏂囦欢鍚嶅墠缂€銆?- `KIND_TO_ONEHOT`锛氳繃娓＄粨鏋?NN 杈撳叆涓殑鍣ㄤ欢绫诲瀷缂栫爜銆?- `TRANSITION_VALUE_NAMES`锛氳繃娓＄粨鏋?NN 杈撳嚭椤哄簭 `[L1, R1, L2, R2, C1, G1]`銆?- 缁樺浘鏇茬嚎鏍峰紡銆?
### `utils.py`

鍩虹宸ュ叿鍑芥暟锛?
- 璺緞澶勭悊锛歚script_base_dir`銆乣as_abs_path`
- `skrf.Network` 鏋勯€狅細`network_from_s`銆乣network_from_abcd`
- HFSS `.s2p` 璇诲彇锛歚load_hfss_network`
- S 鍙傛暟鍜?ABCD 杞崲锛歚s2abcd_np`銆乣abcd2s_np`銆乣abcd2s_torch`
- ABCD 绾ц仈锛歚cascade_abcd_np`

### `io.py`

杈撳叆鏂囦欢瑙ｆ瀽锛?
- `parse_s2p_header_params(filepath)`锛氳鍙?`.s2p` 鏂囦欢寮€澶存敞閲婅涓殑 `key=value` 鍑犱綍鍙傛暟銆?
### `devices.py`

鍣ㄤ欢缁撴瀯瀹氫箟鍜屽嚑浣曞弬鏁扮粍瑁咃細

- `DeviceBlock`锛氬崟涓?RDL/TSV 鍣ㄤ欢鍧楋紝鍖呭惈绫诲瀷銆侀暱搴︺€佸嚑浣曠壒寰併€佺瓑鏁堝弬鏁板拰 RLGC銆?- `make_device_block`锛氭牴鎹?`.s2p` 澶撮儴鍙傛暟鍒涘缓鍗曚釜鍣ㄤ欢鍧椼€?- `build_structure_blocks`锛氭寜 `DEVICE_SEQUENCE` 鍒涘缓瀹屾暣 13 娈靛櫒浠剁粨鏋勩€?- `shortened_length_scales`锛氭彃鍏ヨ繃娓＄粨鏋勫悗锛岃绠楁瘡娈靛櫒浠朵繚鐣欓暱搴︽瘮渚嬨€?
闀垮害缂╂斁瑙勫垯锛?
- 棣栧熬鍣ㄤ欢淇濈暀 `0.9 * Length`
- 涓棿鍣ㄤ欢淇濈暀 `0.8 * Length`
- 琚墸闄ょ殑 `0.1 * Length` 鐢ㄤ簬鐩搁偦杩囨浮缁撴瀯寤烘ā

### `matlab_nn.py`

璋冪敤 MATLAB 瀵煎嚭鐨?`.mat` 绁炵粡缃戠粶锛?
- `predict_one_matlab_nn`锛氳皟鐢ㄥ崟涓?`.mat` 缃戠粶棰勬祴涓€涓數璺弬鏁般€?- `predict_circuit_parameters`锛氬涓€涓櫒浠堕娴嬪叏閮?9 涓瓑鏁堢數璺弬鏁般€?- `attach_circuit_params_to_blocks`锛氫负鎵€鏈夊櫒浠跺潡闄勫姞鐢佃矾鍙傛暟鍜?RLGC銆?
璇ユā鍧楀亣璁?`.mat` 鏂囦欢鍖呭惈锛?
```text
psmin, psmax,
w1, theta1,
w2, theta2,
w3, theta3,
outputmax, outputmin
```

### `circuit.py`

绛夋晥鐢佃矾鍙傛暟鍒扮數纾佺綉缁滃弬鏁扮殑杞崲锛?
- `circuit_params_to_rlgc`锛氭牴鎹瓑鏁堢數璺叕寮忚绠楀崟浣嶉暱搴?`R/L/G/C`銆?- `rlgc_to_abcd`锛氬皢浼犺緭绾?RLGC 妯″瀷杞崲涓?ABCD 鐭╅樀銆?- `block_to_abcd`锛氬皢鍗曚釜 `DeviceBlock` 杞崲涓?ABCD銆?- `block_to_network`锛氬皢鍗曚釜 `DeviceBlock` 杞崲涓?`skrf.Network`銆?
### `transition.py`

杩囨浮缁撴瀯鎻愬彇鍜岀骇鑱旓細

- `transition_values_from_blocks`锛氱敱宸﹀彸鐩搁偦鍣ㄤ欢鐨?`0.1 * Length` RLGC 鎻愬彇杩囨浮缁撴瀯鍏冧欢銆?- `transition_abcd_from_values`锛氬皢 `[L1, R1, L2, R2, C1, G1]` 杞崲涓?ABCD銆?- `build_transition_values_for_structure`锛氫负瀹屾暣缁撴瀯涓墍鏈夌浉閭诲櫒浠剁敓鎴愯繃娓″厓浠躲€?- `cascade_with_transitions_np`锛氭墽琛屸€滅缉鐭櫒浠?+ 杩囨浮缁撴瀯鈥濈殑鏁翠綋绾ц仈銆?
杩囨浮缁撴瀯鎷撴墤锛?
```text
Port1 -- L1 -- R1 -- node -- L2 -- R2 -- Port2
                           |
                         C1 || G1
                           |
                          GND
```

### `model.py`

杩囨浮缁撴瀯绁炵粡缃戠粶鍙婄洃鐫ｈ缁冿細

- `Normalizer`锛氫繚瀛樿緭鍏ョ壒寰佸拰杈撳嚭 log 鍏冧欢鍊肩殑鏍囧噯鍖栧弬鏁般€?- `transition_input_vector`锛氭瀯閫?NN 杈撳叆鐗瑰緛銆?- `build_transition_training_data`锛氱敓鎴愮洃鐫ｈ缁冮泦 `X_raw/Y_raw`銆?- `TransitionElementNN`锛氳繃娓＄粨鏋勫厓浠跺€奸娴嬬綉缁溿€?- `make_normalizer`锛氱敓鎴愭爣鍑嗗寲鍣ㄣ€?- `train_supervised_transition_nn`锛氱敤鎻愬彇鐨勮繃娓＄粨鏋勫厓浠跺€肩洃鐫ｈ缁?NN銆?- `predict_transition_values_np`锛氱敤璁粌濂界殑 NN 棰勬祴瀹屾暣缁撴瀯涓殑鎵€鏈夎繃娓″厓浠躲€?
NN 杈撳叆缁村害涓?17锛?
```text
left_type_onehot(3)
right_type_onehot(3)
left_geom5(5)
right_geom5(5)
freq_GHz(1)
```

NN 杈撳嚭缁村害涓?6锛?
```text
[L1, R1, L2, R2, C1, G1]
```

璁粌鏃惰緭鍑虹洰鏍囦娇鐢?`log(Y)` 鍚庢爣鍑嗗寲锛屼互鍑忓皬涓嶅悓閲忕翰閫犳垚鐨勬暟鍊煎樊寮傘€?
### `torch_cascade.py`

PyTorch 鐗堟湰鐨勮繃娓＄粨鏋勭骇鑱旓紝鐢ㄤ簬绔埌绔井璋冿細

- `transition_abcd_torch`锛氳繃娓＄粨鏋勫厓浠跺€艰浆 ABCD銆?- `cascade_with_transition_values_torch`锛氬彲寰垎绾ц仈骞惰緭鍑?S 鍙傛暟銆?- `fine_tune_transition_nn_on_hfss`锛氬崟 DUT HFSS 鐩爣寰皟鍏ュ彛銆?
澶?DUT 绔埌绔井璋冪殑涓诲嚱鏁板湪 `dataset.py` 涓€?
### `metrics_plot.py`

璇勪及鍜岀粯鍥撅細

- `complex_mse`锛氳绠楀鏁?S 鍙傛暟 MSE銆?- `print_mse_table`锛氭墦鍗板崟涓?DUT 鐨勬ā鍨嬪姣?MSE銆?- `print_dataset_mse_summary`锛氭墦鍗版暟鎹泦 MSE 姹囨€汇€?- `plot_s_comparison`锛氱粯鍒?HFSS銆佺洿鎺ョ骇鑱斻€佹彁鍙栬繃娓°€丯N 鐩戠潱銆丯N 寰皟缁撴灉瀵规瘮銆?
### `persistence.py`

淇濆瓨鍏抽敭涓棿缁撴灉锛屼究浜庡悗缁垎鏋愬拰璋冪敤锛?
- `save_structure_sample`锛氫繚瀛樺崟 DUT 鐨勬牱鏈噯澶囩粨鏋溿€?- `save_training_dataset`锛氫繚瀛樺悎骞跺悗鐨勭洃鐫ｈ缁冮泦銆?- `save_normalizer`锛氫繚瀛樻爣鍑嗗寲鍙傛暟銆?- `save_model_checkpoint`锛氫繚瀛?NN 鏉冮噸鍜屾爣鍑嗗寲鍣ㄣ€?- `save_evaluation_result`锛氫繚瀛樻瘡涓?DUT 鐨勯娴嬬粨鏋溿€丼 鍙傛暟鍜?MSE銆?- `save_mse_summary`锛氫繚瀛樺叏鏁版嵁闆?MSE 姹囨€汇€?
榛樿淇濆瓨鐩綍锛?
```text
model_results/training/RDL_TSV_results/
  intermediate/
    dut001/
      metadata.json
      sample_arrays.npz
      evaluation_arrays.npz
      mse.json
    dut002/
      ...
    dataset/
      transition_training_dataset.npz
      mse_summary.csv
      error_analysis.json
      error_analysis.md
    models/
      transition_normalizer.npz
      transition_model_supervised.pth
      transition_model_fine_tuned.pth
    loss_curves/
      supervised_pretrain_loss.csv
      supervised_pretrain_loss.png
      hfss_fine_tune_loss.csv
      hfss_fine_tune_loss.png
```

### `dataset.py`

瀹屾暣澶?DUT 宸ヤ綔娴佷富妯″潡锛?
- `StructureSample`锛氬崟涓?DUT 鐨勫畬鏁存牱鏈暟鎹粨鏋勩€?- `prepare_structure_sample`锛氬噯澶囦竴涓?DUT 鐨?HFSS銆佸櫒浠跺潡銆丷LGC銆丄BCD銆佽繃娓＄粨鏋勫拰璁粌鏍锋湰銆?- `collect_structure_samples`锛氭敹闆嗗涓?DUT銆?- `evaluate_sample_with_transition_model`锛氱敤杩囨浮缁撴瀯 NN 璇勪及涓€涓?DUT銆?- `fine_tune_transition_nn_on_dataset`锛氫互澶氫釜 DUT 鐨?HFSS S 鍙傛暟涓哄叡鍚岀洰鏍囧井璋冨叡浜?NN銆?- `run_dataset_training`锛氭帹鑽愪富鍏ュ彛銆?- `run_one_dut`锛氬崟 DUT 璋冭瘯鍏ュ彛銆?- `run_batch`锛氬吋瀹规棫鍏ュ彛銆?
## 鏁翠綋宸ヤ綔娴佺▼

瀹屾暣娴佺▼鐢?`run_dataset_training` 椹卞姩銆?
### Step 1锛氭敹闆?DUT 骞舵瀯寤鸿缁冩暟鎹?
瀵?`start_idx..end_idx` 涓瓨鍦ㄧ殑 `dut*.s2p`锛?
1. 璇诲彇 HFSS 鏁翠綋缁撴瀯 S 鍙傛暟銆?2. 瑙ｆ瀽 `.s2p` 澶撮儴鍑犱綍鍙傛暟銆?3. 鎸夊浐瀹氬簭鍒楀垱寤?RDL/TSV 鍣ㄤ欢鍧椼€?4. 璋冪敤 `.mat` 绁炵粡缃戠粶棰勬祴姣忎釜鍣ㄤ欢鐨勭瓑鏁堢數璺弬鏁般€?5. 璁＄畻姣忎釜鍣ㄤ欢鐨勫崟浣嶉暱搴?RLGC銆?6. 鏋勯€犲畬鏁撮暱搴︾洿鎺ョ骇鑱旂粨鏋溿€?7. 鏋勯€犵缉鐭櫒浠舵銆?8. 浠庣浉閭诲櫒浠舵彁鍙栬繃娓＄粨鏋勫厓浠躲€?9. 鐢熸垚鐩戠潱璁粌鏍锋湰 `X_raw/Y_raw`銆?
### Step 2锛氱洃鐫ｈ缁冨叡浜繃娓＄粨鏋?NN

鍚堝苟鎵€鏈?DUT 鐨?`X_raw/Y_raw`锛?
```text
X_all = vstack(sample.X_raw)
Y_all = vstack(sample.Y_raw)
```

浣跨敤鎻愬彇鍑烘潵鐨勮繃娓＄粨鏋勫厓浠跺€间綔涓虹洃鐫ｇ洰鏍囷紝璁粌涓€涓叡浜?`TransitionElementNN`銆?
杈撳嚭锛?
- `transition_model_supervised`
- `transition_normalizer`
- `supervised_loss_history`
- `loss_curves/supervised_pretrain_loss.png`
- `loss_curves/supervised_pretrain_loss.csv`

### Step 3锛欻FSS 绔埌绔井璋?
浠ュ涓?DUT 鐨?HFSS 鏁翠綋 S 鍙傛暟涓哄叡鍚岀洰鏍囷紝缁х画寰皟鍚屼竴涓叡浜?NN銆?
鎹熷け鍑芥暟锛?
```text
loss = mean(MSE(S_pred, S_HFSS))
       + fine_reg_weight * mean(MSE(predicted_transition_norm, extracted_transition_norm))
```

绗簩椤圭敤浜庣害鏉熷井璋冨悗鐨勮繃娓″厓浠朵笉瑕佽繃搴﹀亸绂荤敱 `0.1 * Length` RLGC 鎻愬彇寰楀埌鐨勫垵濮嬬墿鐞嗕及璁°€?
杈撳嚭锛?
- `transition_model_fine_tuned`
- `fine_tune_loss_history`
- `loss_curves/hfss_fine_tune_loss.png`
- `loss_curves/hfss_fine_tune_loss.csv`

### Step 4锛氳瘎浼板拰淇濆瓨缁撴灉

瀵规瘡涓?DUT 杈撳嚭浠ヤ笅妯″瀷瀵规瘮锛?
```text
Direct full cascade
Extracted transition
NN supervised transition
NN fine-tuned transition
```

骞惰绠楃浉瀵?HFSS 鐨勫鏁?S 鍙傛暟 MSE銆?
璁粌瀹屾垚鍚庤繕浼氬鎵€鏈?sample 鐨勮宸仛缁熻鍒嗘瀽锛?
- 璁＄畻姣忎釜妯″瀷鐨?mean銆乵edian銆乻td銆乵in銆乵ax MSE銆?- 鎵惧嚭骞冲潎 MSE 鏈€浼樻ā鍨嬨€?- 鎸夋渶缁堟ā鍨?MSE 瀵?DUT 鎺掑簭锛屽畾浣嶈宸渶澶х殑 sample銆?- 鏍规嵁鐩存帴绾ц仈銆佹彁鍙栬繃娓°€佺洃鐫?NN銆佸井璋?NN 鐨勭浉瀵规敼鍠勬儏鍐电敓鎴愭敼杩涘缓璁€?
鍒嗘瀽缁撴灉淇濆瓨鍒帮細

```text
model_results/training/RDL_TSV_results/intermediate/dataset/error_analysis.json
model_results/training/RDL_TSV_results/intermediate/dataset/error_analysis.md
```

濡傛灉寮€鍚?`plot=True` 鎴?`save_plot=True`锛屼細缁樺埗锛?
- S11 magnitude
- S21 magnitude
- S11 phase
- S21 phase

## 浣跨敤鏂规硶

### 鏂瑰紡 1锛氳繍琛屽吋瀹瑰叆鍙ｈ剼鏈?
鍦?`Temp` 鐩綍涓嬭繍琛岋細

```bash
python rdl_tsv_transition_dataset_train.py
```

璇ュ叆鍙ｄ娇鐢ㄩ粯璁ゅ弬鏁帮細

```python
run_dataset_training(
    start_idx=1,
    end_idx=10,
    s2p_dir="./snp_data/RDL_TSV_Snp",
    mat_dir="./device_models/RDL_TSV_mat2",
    max_points=None,
    supervised_epochs=2000,
    fine_epochs=1000,
    supervised_lr=2e-3,
    fine_lr=2e-4,
    fine_reg_weight=1e-4,
    hidden=128,
    supervised_batch_size=8192,
    fine_sample_batch_size=2,
    plot=True,
    save_plot=False,
    out_dir="./model_results/training/RDL_TSV_results",
    save_intermediate=True,
    verbose=True,
)
```

### 鏂瑰紡 2锛氬湪 Python 涓皟鐢?
```python
from rdl_tsv_transition import run_dataset_training

output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    s2p_dir="./snp_data/RDL_TSV_Snp",
    mat_dir="./device_models/RDL_TSV_mat2",
    max_points=300,
    supervised_epochs=500,
    fine_epochs=200,
    plot=False,
    save_plot=True,
    out_dir="./model_results/training/RDL_TSV_results",
    save_intermediate=True,
)
```

杩斿洖鍊兼槸涓€涓瓧鍏革細

```python
{
    "samples": samples,
    "results": results,
    "transition_model_supervised": model_supervised,
    "transition_model_fine_tuned": model_fine_tuned,
    "transition_normalizer": normalizer,
    "mse_rows": mse_rows,
    "supervised_loss_history": supervised_loss_history,
    "fine_tune_loss_history": fine_tune_loss_history,
    "error_analysis": error_analysis,
}
```

### 鍗?DUT 璋冭瘯

```python
from rdl_tsv_transition import run_one_dut

result = run_one_dut(
    idx=1,
    s2p_dir="./snp_data/RDL_TSV_Snp",
    mat_dir="./device_models/RDL_TSV_mat2",
    max_points=300,
    supervised_epochs=200,
    fine_epochs=100,
    plot=False,
)
```

### 璺宠繃绔埌绔井璋?
```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    fine_epochs=0,
)
```

### 涓嶄繚瀛樹腑闂寸粨鏋?
```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    save_intermediate=False,
)
```

### 鍙繚瀛樺浘锛屼笉寮瑰嚭鍥剧獥

```python
output = run_dataset_training(
    start_idx=1,
    end_idx=10,
    plot=False,
    save_plot=True,
)
```

## 涓棿缁撴灉璇诲彇绀轰緥

璇诲彇鍚堝苟鍚庣殑鐩戠潱璁粌闆嗭細

```python
import numpy as np

data = np.load("./model_results/training/RDL_TSV_results/intermediate/dataset/transition_training_dataset.npz")
X_all = data["X_all"]
Y_all = data["Y_all"]
```

璇诲彇鏌愪釜 DUT 鐨勬牱鏈暟缁勶細

```python
sample = np.load("./model_results/training/RDL_TSV_results/intermediate/dut001/sample_arrays.npz")
freqs_hz = sample["freqs_hz"]
hfss_s = sample["hfss_s"]
direct_full_s = sample["direct_full_s"]
extracted_transition_s = sample["extracted_transition_s"]
X_raw = sample["X_raw"]
Y_raw = sample["Y_raw"]
```

璇诲彇妯″瀷锛?
```python
import torch
from rdl_tsv_transition.model import TransitionElementNN, Normalizer

ckpt = torch.load("./model_results/training/RDL_TSV_results/intermediate/models/transition_model_fine_tuned.pth")

model = TransitionElementNN(hidden=ckpt["extra"]["hidden"]).to(dtype=torch.float64)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

n = ckpt["normalizer"]
normalizer = Normalizer(
    x_mean=n["x_mean"],
    x_std=n["x_std"],
    y_mean=n["y_mean"],
    y_std=n["y_std"],
)
```

璇诲彇 loss 鍘嗗彶锛?
```python
import pandas as pd

pretrain_loss = pd.read_csv("./model_results/training/RDL_TSV_results/intermediate/loss_curves/supervised_pretrain_loss.csv")
fine_loss = pd.read_csv("./model_results/training/RDL_TSV_results/intermediate/loss_curves/hfss_fine_tune_loss.csv")
```

璇诲彇璇樊鍒嗘瀽锛?
```python
import json

with open("./model_results/training/RDL_TSV_results/intermediate/dataset/error_analysis.json", "r", encoding="utf-8") as f:
    analysis = json.load(f)

print(analysis["best_model_by_mean_mse"])
print(analysis["recommendations"])
```

## 甯歌鍙傛暟璇存槑

| 鍙傛暟 | 璇存槑 |
| --- | --- |
| `start_idx`, `end_idx` | DUT 缂栧彿鑼冨洿锛屽搴?`dut{idx}.s2p` |
| `s2p_dir` | HFSS 鏁翠綋缁撴瀯 `.s2p` 鐩綍 |
| `mat_dir` | MATLAB 瀵煎嚭鐨勫櫒浠剁骇 `.mat` 缃戠粶鍙傛暟鐩綍 |
| `max_points` | 鍙娇鐢ㄥ墠 N 涓鐐癸紱`None` 琛ㄧず浣跨敤鍏ㄩ儴棰戠偣 |
| `supervised_epochs` | 杩囨浮缁撴瀯 NN 鐩戠潱璁粌杞暟 |
| `fine_epochs` | HFSS 绔埌绔井璋冭疆鏁帮紱璁句负 0 鍙烦杩?|
| `supervised_lr` | 鐩戠潱璁粌瀛︿範鐜?|
| `fine_lr` | 绔埌绔井璋冨涔犵巼 |
| `fine_reg_weight` | 寰皟鏃剁害鏉熻繃娓″厓浠跺亸绂绘彁鍙栧€肩殑姝ｅ垯鏉冮噸 |
| `hidden` | 杩囨浮缁撴瀯 NN 闅愯棌灞傚搴?|
| `supervised_batch_size` | 鐩戠潱璁粌 batch size |
| `fine_sample_batch_size` | 绔埌绔井璋冩椂姣忎釜 batch 鍖呭惈鐨?DUT 鏁伴噺 |
| `plot` | 鏄惁鏄剧ず瀵规瘮鍥?|
| `save_plot` | 鏄惁淇濆瓨瀵规瘮鍥?|
| `out_dir` | 杈撳嚭鐩綍 |
| `save_intermediate` | 鏄惁淇濆瓨鍏抽敭涓棿缁撴灉 |
| `verbose` | 鏄惁鎵撳嵃璇︾粏璁粌鏃ュ織 |

## 娉ㄦ剰浜嬮」

1. `s2p_dir` 鍜?`mat_dir` 鐨勭浉瀵硅矾寰勫熀鍑嗘槸鍏ュ彛鑴氭湰鎵€鍦ㄧ洰褰曘€?2. `.mat` 鏂囦欢鍚嶅繀椤绘弧瓒?`MAT_PREFIX + CIRCUIT_PARAM_NAME + ".mat"` 鐨勮鍒欍€?3. `TSV` 鐨勫嚑浣曡緭鍏ュ彧鏈?3 缁达紝浠ｇ爜浼?padding 鍒?5 缁达紝骞堕€氳繃 one-hot 绫诲瀷鍖哄垎鍚箟銆?4. 璁粌鍜岀骇鑱旈粯璁や娇鐢?`float64/complex128`锛屼互闄嶄綆楂橀绾ц仈鏃剁殑鏁板€艰宸€?5. 濡傛灉鏁版嵁鐐瑰緢澶氫笖 GPU/鍐呭瓨涓嶈冻锛屼紭鍏堝噺灏?`max_points`銆乣supervised_batch_size` 鎴?`fine_sample_batch_size`銆?
