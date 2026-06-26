import os
from pathlib import Path
import re
import numpy as np
import scipy.io as sio
import skrf as rf
import matplotlib.pyplot as plt
import glob

# ==========================================
# 1. 浠?s2p 鏂囦欢涓鍙栧櫒浠剁墿鐞嗗弬鏁?
# ==========================================
def extract_device_params_RDL_Top(filepath):
    """
    瑙ｆ瀽 .s2p 鏂囦欢澶撮儴鐨勬敞閲婏紝鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟 (浣滀负 NN 鐨勮緭鍏?x)
    鍏煎 RDL_top (lrdl) 鍜?RDL_bottom (ldown)
    """
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): 
                break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
                        
    # 鍔ㄦ€侀€傞厤 lrdl(Top) 鎴?ldown(Bottom)
    try:
        length = params.get('lrdl')
        width = params.get('wrdl')
        thickness = params.get('trdl')
        htsv = params['htsv']
        p1 = params['p1']
        
        features = np.array([length, width, thickness, htsv, p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 鏂囦欢缂哄皯蹇呰鐨勫弬鏁版敞閲? {e}")
    

def extract_device_params_RDL_Bottom(filepath):
    """
    瑙ｆ瀽 .s2p 鏂囦欢澶撮儴鐨勬敞閲婏紝鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟 (浣滀负 NN 鐨勮緭鍏?x)
    鍏煎 RDL_top (lrdl) 鍜?RDL_bottom (ldown)
    """
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): 
                break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
                        
    # 鍔ㄦ€侀€傞厤 lrdl(Top) 鎴?ldown(Bottom)
    try:
        length = params.get('ldown')
        width = params.get('wdown')
        thickness = params.get('tdown')
        htsv = params['htsv']
        p1 = params['p1']
        
        features = np.array([length, width, thickness, htsv, p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 鏂囦欢缂哄皯蹇呰鐨勫弬鏁版敞閲? {e}")

def extract_device_params_TSV(filepath):
    """
    瑙ｆ瀽 .s2p 鏂囦欢澶撮儴鐨勬敞閲婏紝鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟 (浣滀负 NN 鐨勮緭鍏?x)
    鍏煎 RDL_top (lrdl) 鍜?RDL_bottom (ldown)
    """
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): 
                break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
                        
    # 鍔ㄦ€侀€傞厤 lrdl(Top) 鎴?ldown(Bottom)
    try:
        dtsv = params.get('dtsv')
        length = params.get('htsv')
        p1 = params['p1']
        
        features = np.array([dtsv,length,p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 鏂囦欢缂哄皯蹇呰鐨勫弬鏁版敞閲? {e}")
    

def extract_device_params_RDL_TSV(filepath):
    """
    瑙ｆ瀽 ./RDL_TSV/dut{idx}.s2p 鏁翠綋閾捐矾鏂囦欢澶撮儴鐨勬敞閲娿€?
    鎻愬彇鍑哄寘鍚?Top銆丅ottom 鍜?TSV 鐨勬墍鏈夌墿鐞嗗弬鏁般€?
    """
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): 
                break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    return params

# ==========================================
# 2. 绁炵粡缃戠粶鎺ㄧ悊 (寰幆鍔犺浇 9 涓?.mat 鏂囦欢)
# ==========================================
def predict_circuit_parameters(features, mat_dir, param_names, prefix="RDL_Bottom_"):
    """
    璇诲彇 MATLAB 瀵煎嚭鐨勬潈鍊硷紝棰勬祴鍑?9 涓瓑鏁堢數璺缉鏀惧弬鏁?(nH, pF, Ohm)
    """
    circuit_params = {}
    x = features.reshape(1, -1)
    
    for param in param_names:
        mat_filepath = os.path.join(mat_dir, f"{prefix}{param}.mat")
        
        if not os.path.exists(mat_filepath):
            print(f"Warning: model file not found: {mat_filepath}; using safe default.")
            circuit_params[param] = 1.0 
            continue
            
        mat_data = sio.loadmat(mat_filepath)
        xmin = mat_data['psmin']
        xmax = mat_data['psmax']
        ymin = mat_data['outputmin']
        ymax = mat_data['outputmax']
        
        w1, b1 = mat_data['w1'], mat_data['theta1']
        w2, b2 = mat_data['w2'], mat_data['theta2']
        w3, b3 = mat_data['w3'], mat_data['theta3']
        
        # 褰掍竴鍖?[-1, 1]
        x_norm = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        
        # 鍓嶅悜浼犳挱 (鍙岄殣钘忓眰 tansig -> purelin)
        a1 = np.tanh(np.dot(x_norm, w1) + b1)
        a2 = np.tanh(np.dot(a1, w2) + b2)
        y_norm = np.dot(a2, w3) + b3
        
        # 鍙嶅綊涓€鍖?
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[param] = float(y_real.flatten()[0])
        
    return circuit_params

# ==========================================
# 3. RLGC -> 璁＄畻 S 鍙傛暟 (瀹岀編瀵归綈鎮ㄧ殑棰戝彉鏁板妯″瀷)
# ==========================================
def calculate_S_parameters(circuit_params, length_um, freqs):
    """
    1. 鍙嶇缉鏀剧綉缁滆緭鍑虹殑 nH 鍜?pF 鍙傛暟鍒版爣鍑嗗崟浣?(H, F)銆?
    2. 璁＄畻闅忛鐜囧彉鍖栫殑鍗曚綅闀垮害鍒嗗竷鍙傛暟 (R_RLGC, L_RLGC, G_RLGC, C_RLGC)銆?
    3. 鏍规嵁闀垮害杞寲涓?ABCD 鐭╅樀骞舵彁鍙?S 鐭╅樀銆?
    """
    # 鎻愬彇骞堕€嗙缉鏀剧綉缁滃弬鏁帮細L 鎭㈠涓?H (涔?1e-9)锛孋 鎭㈠涓?F (涔?1e-12)
    R1 = circuit_params["R1"]
    R2 = circuit_params["R2"]
    R3 = circuit_params["R3"]
    L1 = circuit_params["L1"] * 1e-9
    L2 = circuit_params["L2"] * 1e-9
    L3 = circuit_params["L3"] * 1e-9
    Cox = circuit_params["Cox"] * 1e-12
    Csi = circuit_params["Csi"] * 1e-12
    Rsi = circuit_params["Rsi"]
    
    # 鐗╃悊闀垮害杞崲 (um -> m)
    length_m = length_um * 1e-6 
    omega = 2 * np.pi * freqs
    
    # === 浣跨敤 NumPy 鍚戦噺鍖栨搷浣滃姞閫熼鍙樺叕寮忚绠?(涓庢彁鍙?.py瀹屽叏涓€鑷? ===
    # 涓茶仈闃绘姉鏀矾 (瓒嬭偆涓庢丁娴?
    R_RLGC = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / ((R1 + R2)**2 + omega**2 * L2**2) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_RLGC = (R1**2 * L2) / ((R1 + R2)**2 + omega**2 * L2**2) + L3 * R3**2 / (R3**2 + omega**2 * L3**2) + L1
    
    # 骞惰仈瀵肩撼鏀矾 (纭呰‖搴曡壊鏁?
    G_RLGC = (omega**2 * Rsi * Cox**2) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)
    C_RLGC = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)

    # 璁＄畻鐗瑰緛闃绘姉涓庝紶鎾父鏁?
    Z0 = np.sqrt((R_RLGC + 1j * omega * L_RLGC) / (G_RLGC + 1j * omega * C_RLGC))
    GAMMA = np.sqrt((R_RLGC + 1j * omega * L_RLGC) * (G_RLGC + 1j * omega * C_RLGC))

    # 浼犺緭绾跨殑 ABCD 鍙傛暟
    A = np.cosh(GAMMA * length_m)
    B = Z0 * np.sinh(GAMMA * length_m)
    C_mat = (1 / Z0) * np.sinh(GAMMA * length_m)
    D = np.cosh(GAMMA * length_m)
    
    # ABCD 杞?S 鍙傛暟鍏紡
    denom = A + B/50.0 + C_mat*50.0 + D
    S11 = (A + B/50.0 - C_mat*50.0 - D) / denom
    S12 = 2 * (A*D - B*C_mat) / denom
    S21 = 2 / denom
    S22 = (-A + B/50.0 - C_mat*50.0 + D) / denom
    
    # 缁勫悎涓?S 鐭╅樀缁? (棰戠偣鏁? 2, 2)
    S_matrices = np.zeros((len(freqs), 2, 2), dtype=complex)
    S_matrices[:, 0, 0] = S11
    S_matrices[:, 0, 1] = S12
    S_matrices[:, 1, 0] = S21
    S_matrices[:, 1, 1] = S22
        
    return S_matrices

# ==========================================
# 4. 鍙鍖栧姣?
# ==========================================
def Plot_S_Comparison(hfss_nw, nn_nw):
    # 5. 鍙鍖栧姣?
    plt.figure(figsize=(12, 5))
    
    # 鐢?S11 骞呭害
    plt.subplot(1, 2, 1)
    hfss_nw.plot_s_db(m=0, n=0, color='blue', label='HFSS $S_{11}$')
    nn_nw.plot_s_db(m=0, n=0, color='red', linestyle='--', label='NN $S_{11}$')
    plt.title("S11 Magnitude (dB)")
    plt.grid(True)
    
    # 鐢?S21 骞呭害
    plt.subplot(1, 2, 2)
    hfss_nw.plot_s_db(m=1, n=0, color='blue', label='HFSS $S_{21}$')
    nn_nw.plot_s_db(m=1, n=0, color='red', linestyle='--', label='NN $S_{21}$')
    plt.title("S21 Magnitude (dB)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    mse = np.mean(np.abs(hfss_nw.s - nn_nw.s)**2)
    print(f"\n>>> 瀵规瘮缁撴潫锛佹暣浣?S 鐭╅樀鍧囨柟璇樊 (MSE): {mse:.4e}")


# ==========================================
# 5. 璁＄畻S鍙傛暟骞舵瀯閫犵綉缁?
# ==========================================

def Calc_RDL_Top_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 鍒囨崲鍒板綋鍓嶈剼鏈洰褰曪紝纭繚璺緞姝ｇ‘
    s2p_file = rf"./snp_data/RDL_Top_Snp/dut{idx}.s2p"         # HFSS 鍘熷娴嬭瘯鏁版嵁
    mat_dir  = r"./device_models/RDL_TSV_mat2"                            # .mat 妯″瀷瀛樻斁鐨勭洰褰?
    model_prefix = "RDL_Top_"                # 绁炵粡缃戠粶瀵煎嚭鐨勫墠缂€
    
    if not os.path.exists(s2p_file):
        print(f"鏈壘鍒版祴璇曟枃浠? {s2p_file}")
        return

    # 1. 鎻愬彇 HFSS 鐪熷疄 S 鍙傛暟
    print(">>> 姝ｅ湪鍔犺浇 HFSS 鍘熷 S鍙傛暟鏂囦欢...")
    RDL_Top_HFSS_NW = rf.Network(s2p_file)
    freqs = RDL_Top_HFSS_NW.f          
    
    # 2. 鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟浣滀负杈撳叆
    device_features, length_um = extract_device_params_RDL_Top(s2p_file)
    print(device_features)
    print(f">>> 鎻愬彇鍒扮墿鐞嗙壒寰佸悜閲? {device_features}")
    
    # 3. 寰幆杞藉叆 9 涓缁忕綉缁滄ā鍨嬮娴嬪弬鏁?
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 姝ｅ湪寰幆鎺ㄧ悊绁炵粡缃戠粶锛岄娴嬬瓑鏁堢數璺弬鏁?..")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 绛夋晥鐢佃矾杞寲涓洪娴嬬殑 S 鐭╅樀
    print(">>> 姝ｅ湪鎭㈠鐗╃悊閲忕骇锛屽苟鍩轰簬 RLGC 棰戝彉妯″瀷鐢熸垚 S 鍙傛暟...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    RDL_Top_NN_NW = rf.Network(frequency=RDL_Top_HFSS_NW.frequency, s=predicted_s_matrices, name="RDL_Top_NN_Predicted")
    RDL_Top_HFSS_NW.name = "RDL_Top_HFSS_Simulated"

    Plot_S_Comparison(RDL_Top_HFSS_NW, RDL_Top_NN_NW)

def Calc_RDL_Bottom_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 鍒囨崲鍒板綋鍓嶈剼鏈洰褰曪紝纭繚璺緞姝ｇ‘
    s2p_file = rf"./snp_data/RDL_Bottom_Snp/dut{idx}.s2p"         # HFSS 鍘熷娴嬭瘯鏁版嵁
    mat_dir  = r"./device_models/RDL_TSV_mat2"                            # .mat 妯″瀷瀛樻斁鐨勭洰褰?
    model_prefix = "RDL_Bottom_"                # 绁炵粡缃戠粶瀵煎嚭鐨勫墠缂€
    
    if not os.path.exists(s2p_file):
        print(f"鏈壘鍒版祴璇曟枃浠? {s2p_file}")
        return

    # 1. 鎻愬彇 HFSS 鐪熷疄 S 鍙傛暟
    print(">>> 姝ｅ湪鍔犺浇 HFSS 鍘熷 S鍙傛暟鏂囦欢...")
    RDL_Bottom_HFSS_NW = rf.Network(s2p_file)
    freqs = RDL_Bottom_HFSS_NW.f          
    
    # 2. 鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟浣滀负杈撳叆
    device_features, length_um = extract_device_params_RDL_Bottom(s2p_file)
    print(device_features)
    print(f">>> 鎻愬彇鍒扮墿鐞嗙壒寰佸悜閲? {device_features}")
    
    # 3. 寰幆杞藉叆 9 涓缁忕綉缁滄ā鍨嬮娴嬪弬鏁?
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 姝ｅ湪寰幆鎺ㄧ悊绁炵粡缃戠粶锛岄娴嬬瓑鏁堢數璺弬鏁?..")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 绛夋晥鐢佃矾杞寲涓洪娴嬬殑 S 鐭╅樀
    print(">>> 姝ｅ湪鎭㈠鐗╃悊閲忕骇锛屽苟鍩轰簬 RLGC 棰戝彉妯″瀷鐢熸垚 S 鍙傛暟...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    
    RDL_Bottom_NN_NW = rf.Network(frequency=RDL_Bottom_HFSS_NW.frequency, s=predicted_s_matrices, name="RDL_Bottom_NN_Predicted")
    RDL_Bottom_HFSS_NW.name = "RDL_Bottom_HFSS_Simulated"
    Plot_S_Comparison(RDL_Bottom_HFSS_NW, RDL_Bottom_NN_NW)

def Calc_TSV_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 鍒囨崲鍒板綋鍓嶈剼鏈洰褰曪紝纭繚璺緞姝ｇ‘
    s2p_file = rf"./snp_data/TSV_Snp/dut{idx}.s2p"         # HFSS 鍘熷娴嬭瘯鏁版嵁
    mat_dir  = r"./device_models/RDL_TSV_mat2"                            # .mat 妯″瀷瀛樻斁鐨勭洰褰?
    model_prefix = "TSV_"                # 绁炵粡缃戠粶瀵煎嚭鐨勫墠缂€
    
    if not os.path.exists(s2p_file):
        print(f"鏈壘鍒版祴璇曟枃浠? {s2p_file}")
        return

    # 1. 鎻愬彇 HFSS 鐪熷疄 S 鍙傛暟
    print(">>> 姝ｅ湪鍔犺浇 HFSS 鍘熷 S鍙傛暟鏂囦欢...")
    TSV_HFSS_NW = rf.Network(s2p_file)
    freqs = TSV_HFSS_NW.f          
    
    # 2. 鎻愬彇鍣ㄤ欢鐗╃悊鍙傛暟浣滀负杈撳叆
    device_features, length_um = extract_device_params_TSV(s2p_file)
    print(device_features)
    print(f">>> 鎻愬彇鍒扮墿鐞嗙壒寰佸悜閲? {device_features}")
    
    # 3. 寰幆杞藉叆 9 涓缁忕綉缁滄ā鍨嬮娴嬪弬鏁?
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 姝ｅ湪寰幆鎺ㄧ悊绁炵粡缃戠粶锛岄娴嬬瓑鏁堢數璺弬鏁?..")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 绛夋晥鐢佃矾杞寲涓洪娴嬬殑 S 鐭╅樀
    print(">>> 姝ｅ湪鎭㈠鐗╃悊閲忕骇锛屽苟鍩轰簬 RLGC 棰戝彉妯″瀷鐢熸垚 S 鍙傛暟...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    TSV_NN_NW = rf.Network(frequency=TSV_HFSS_NW.frequency, s=predicted_s_matrices, name="TSV_NN_Predicted")
    TSV_HFSS_NW.name = "TSV_HFSS_Simulated"

    Plot_S_Comparison(TSV_HFSS_NW, TSV_NN_NW)

# ==========================================
# 5. 銆愭柊澧炪€戠骇鑱旇绠楁牳蹇冨嚱鏁?
# ==========================================
def Calc_Cascaded_RDL_TSV_S(idx):
    os.chdir(Path(__file__).resolve().parents[2]) 
    
    # 鎸囧悜闀块摼璺叏绾ц仈鐨?HFSS 娴嬭瘯鏂囦欢
    s2p_file = rf"./snp_data/RDL_TSV_Snp/dut{idx}.s2p"         
    mat_dir  = r"./device_models/RDL_TSV_mat2"                   
    
    if not os.path.exists(s2p_file):
        print(f"鏈壘鍒版祴璇曟枃浠? {s2p_file}")
        return

    # 1. 鎻愬彇 HFSS 鐪熷疄 S 鍙傛暟
    print(">>> 姝ｅ湪鍔犺浇 HFSS 鍘熷鍏ㄩ摼璺?S鍙傛暟鏂囦欢...")
    Cascaded_HFSS_NW = rf.Network(s2p_file)
    freqs = Cascaded_HFSS_NW.f          
    
    # 2. 浠庢€讳綋 s2p 鏂囦欢涓彁鍙栨墍鏈夌墿鐞嗗昂瀵稿弬鏁?
    params = extract_device_params_RDL_TSV(s2p_file)
    print(f">>> 鎻愬彇鍒扮墿鐞嗗弬鏁? {params}")
    
    # 灏嗘彁鍙栧嚭鐨勫弬鏁板垎鍒墦鍖呯粰瀵瑰簲鐨?NN 棰勬祴妯″瀷
    features_top = np.array([params['lrdl'], params['wrdl'], params['trdl'], params['htsv'], params['p1']])
    features_bot = np.array([params['ldown'], params['wdown'], params['tdown'], params['htsv'], params['p1']])
    features_tsv = np.array([params['dtsv'], params['htsv'], params['p1']])
    
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    # 3. 鍒嗗埆璁＄畻 RDL_Top, RDL_Bottom, TSV 鐨勫崟浣?S 鍙傛暟
    print(">>> 姝ｅ湪棰勬祴鍗曚綋缁勪欢鐨勭瓑鏁堢數璺苟璁＄畻 S 鍙傛暟...")
    
    # -- RDL Top --
    cp_top = predict_circuit_parameters(features_top, mat_dir, target_params, prefix="RDL_Top_")
    s_top = calculate_S_parameters(cp_top, params['lrdl'], freqs)
    nw_top = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_top)
    
    # -- RDL Bottom --
    cp_bot = predict_circuit_parameters(features_bot, mat_dir, target_params, prefix="RDL_Bottom_")
    s_bot = calculate_S_parameters(cp_bot, params['ldown'], freqs)
    nw_bot = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_bot)
    
    # -- TSV --
    cp_tsv = predict_circuit_parameters(features_tsv, mat_dir, target_params, prefix="TSV_")
    s_tsv = calculate_S_parameters(cp_tsv, params['htsv'], freqs)
    nw_tsv = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_tsv)
    
    # 4. 鎵ц绾ц仈鐭╅樀杩愮畻 (skrf ** 鎿嶄綔绗︿細鑷姩灏哠杞负T锛岀浉涔樺悗杞洖S)
    print(">>> 姝ｅ湪鎵ц S 鍙傛暟鐭╅樀绾ц仈 (Top - TSV - Bot - TSV - Top - TSV - Bot - TSV - Top) ...")
    
    # 9 娈电骇鑱旇绠?
    nw_cascade = nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top
    
    nw_cascade.name = "9-Stage_Cascaded_NN_Predicted"
    Cascaded_HFSS_NW.name = "RDL_TSV_HFSS_Simulated"

    # 5. 鍙鍖栧姣旂粨鏋?
    Plot_S_Comparison(Cascaded_HFSS_NW, nw_cascade)
    


def Batch_Calc_Cascaded_RDL_TSV_S():
    os.chdir(Path(__file__).resolve().parents[2]) 
    
    # 璺緞閰嶇疆
    input_dir  = r"./snp_data/RDL_TSV_Snp"
    output_dir = r"./snp_data/RDL_TSV_NN_Snp"
    mat_dir    = r"./device_models/RDL_TSV_mat2"
    
    # 濡傛灉杈撳嚭鏂囦欢澶逛笉瀛樺湪锛屽垯鑷姩鍒涘缓
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f">>> 宸插垱寤鸿緭鍑虹洰褰? {output_dir}")

    # 鑾峰彇杈撳叆鐩綍涓嬫墍鏈夌殑 .s2p 鏂囦欢
    s2p_files = glob.glob(os.path.join(input_dir, "*.s2p"))
    
    if not s2p_files:
        print(f"No .s2p files found in {input_dir}.")
        return

    print(f">>> 鍏辨壂鎻忓埌 {len(s2p_files)} 涓?.s2p 鏂囦欢锛屽噯澶囧紑濮嬫壒閲忕骇鑱斿鐞?..\n")

    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]

    # 閬嶅巻澶勭悊姣忎竴涓?s2p 鏂囦欢
    for s2p_file in s2p_files:
        filename = os.path.basename(s2p_file)  # 鑾峰彇璇稿 "dut0.s2p" 鐨勭函鏂囦欢鍚?
        print(f"--- 姝ｅ湪澶勭悊: {filename} ---")
        
        # 1. 鎻愬彇 HFSS 鐪熷疄 S 鍙傛暟锛堜富瑕佺敤浜庤幏鍙栨纭殑棰戠巼鐐?frequency锛?
        try:
            Cascaded_HFSS_NW = rf.Network(s2p_file)
            freqs = Cascaded_HFSS_NW.f          
        except Exception as e:
            print(f"璇诲彇鏂囦欢澶辫触锛岃烦杩?{filename}銆傛姤閿? {e}")
            continue
        
        # 2. 浠庢€讳綋 s2p 鏂囦欢涓彁鍙栨墍鏈夌墿鐞嗗昂瀵稿弬鏁?
        try:
            params = extract_device_params_RDL_TSV(s2p_file)
        except Exception as e:
            print(f"鎻愬彇鐗╃悊鍙傛暟澶辫触锛岃烦杩?{filename}銆傛姤閿? {e}")
            continue
            
        # 灏嗘彁鍙栧嚭鐨勫弬鏁板垎鍒墦鍖呯粰瀵瑰簲鐨?NN 棰勬祴妯″瀷
        features_top = np.array([params['lrdl'], params['wrdl'], params['trdl'], params['htsv'], params['p1']])
        features_bot = np.array([params['ldown'], params['wdown'], params['tdown'], params['htsv'], params['p1']])
        features_tsv = np.array([params['dtsv'], params['htsv'], params['p1']])
        
        # 3. 鍒嗗埆璁＄畻 RDL_Top, RDL_Bottom, TSV 鐨勫崟浣?S 鍙傛暟
        # -- RDL Top --
        cp_top = predict_circuit_parameters(features_top, mat_dir, target_params, prefix="RDL_Top_")
        s_top = calculate_S_parameters(cp_top, params['lrdl'], freqs)
        nw_top = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_top)
        
        # -- RDL Bottom --
        cp_bot = predict_circuit_parameters(features_bot, mat_dir, target_params, prefix="RDL_Bottom_")
        s_bot = calculate_S_parameters(cp_bot, params['ldown'], freqs)
        nw_bot = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_bot)
        
        # -- TSV --
        cp_tsv = predict_circuit_parameters(features_tsv, mat_dir, target_params, prefix="TSV_")
        s_tsv = calculate_S_parameters(cp_tsv, params['htsv'], freqs)
        nw_tsv = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=s_tsv)
        
        # 4. 鎵ц绾ц仈鐭╅樀杩愮畻 
        # (Top - TSV - Bot - TSV - Top - TSV - Bot - TSV - Top)
        nw_cascade = nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top
        
        # 璁剧疆鐢熸垚鐨?Network 鍚嶇О锛岄槻姝㈠鍑虹殑 touchstone 鏂囦欢閲屾湁濂囨€殑娉ㄩ噴鍚?
        nw_cascade.name = filename.replace(".s2p", "")

        # 5. 淇濆瓨绾ц仈寰楀埌鐨?S 鍙傛暟鍒扮洰鏍囩洰褰曪紝鍚嶅瓧淇濇寔涓€涓€瀵瑰簲
        nw_cascade.write_touchstone(filename=filename, dir=output_dir)
        print(f"    [鎴愬姛] 宸茬骇鑱斿苟淇濆瓨鑷?-> {os.path.join(output_dir, filename)}\n")
        

def Compare_Snp_Directories(dir_original="./snp_data/RDL_TSV_Snp", dir_predicted="./snp_data/RDL_TSV_NN_Snp", plot_worst_case=True):
    """
    瀵规瘮涓や釜鏂囦欢澶逛笅鍚屽悕鐨?S 鍙傛暟鏂囦欢銆?
    :param dir_original: 鍘熷 HFSS S鍙傛暟鏂囦欢澶硅矾寰?
    :param dir_predicted: 绁炵粡缃戠粶棰勬祴 S鍙傛暟鏂囦欢澶硅矾寰?
    :param plot_worst_case: 鏄惁鍦ㄥ姣旂粨鏉熷悗鑷姩鐢诲嚭璇樊鏈€澶х殑涓€缁勬暟鎹繘琛屼汉宸ョ‘璁?
    """
    os.chdir(Path(__file__).resolve().parents[2])
    
    if not os.path.exists(dir_original):
        print(f"閿欒: 鎵句笉鍒板師濮嬫枃浠跺す {dir_original}")
        return
    if not os.path.exists(dir_predicted):
        print(f"閿欒: 鎵句笉鍒伴娴嬫枃浠跺す {dir_predicted}")
        return

    # 鑾峰彇鍘熷鏂囦欢澶逛腑鎵€鏈夌殑 .s2p 鏂囦欢
    s2p_files_orig = glob.glob(os.path.join(dir_original, "*.s*p"))
    
    if not s2p_files_orig:
        print(f"No S-parameter files found in {dir_original}.")
        return

    print("==================================================")
    print(f"寮€濮嬫壒閲忓姣?S 鍙傛暟: ")
    print(f"  鍩哄噯鏁版嵁: {dir_original}")
    print(f"  寰呮祴鏁版嵁: {dir_predicted}")
    print("==================================================")

    mse_list = []
    matched_count = 0
    worst_mse = -1
    worst_file = None

    # 閬嶅巻鍘熷鏂囦欢澶归噷鐨勬枃浠?
    for orig_filepath in s2p_files_orig:
        filename = os.path.basename(orig_filepath)
        pred_filepath = os.path.join(dir_predicted, filename)
        
        # 妫€鏌ラ娴嬫枃浠跺す涓槸鍚﹀瓨鍦ㄥ悓鍚嶆枃浠?
        if not os.path.exists(pred_filepath):
            print(f"[璺宠繃] 缂哄け瀵瑰簲鏂囦欢: {filename}")
            continue
            
        try:
            nw_orig = rf.Network(orig_filepath)
            nw_pred = rf.Network(pred_filepath)
            
            # 璁＄畻璇ュ鏂囦欢鐨?MSE (鍧囨柟璇樊)
            mse = np.mean(np.abs(nw_orig.s - nw_pred.s)**2)
            mse_list.append(mse)
            matched_count += 1
            
            # 璁板綍璇樊鏈€澶х殑涓€缁?
            if mse > worst_mse:
                worst_mse = mse
                worst_file = filename
                worst_nw_orig = nw_orig
                worst_nw_pred = nw_pred
                
        except Exception as e:
            print(f"[鎶ラ敊] 澶勭悊鏂囦欢 {filename} 鏃跺彂鐢熷紓甯? {e}")

    # ================= 缁熻涓庢姤鍛?=================
    if matched_count == 0:
        print("No matching file pairs found.")
        return

    mse_array = np.array(mse_list)
    avg_mse = np.mean(mse_array)
    max_mse = np.max(mse_array)
    min_mse = np.min(mse_array)

    print(f"\n瀵规瘮瀹屾垚锛佹垚鍔熼厤瀵规枃浠舵暟: {matched_count}")
    print(f"----------------------------------------")
    print(f"  [鏁翠綋骞冲潎璇樊 (Avg MSE)] : {avg_mse:.6e}")
    print(f"  [鏈€灏忚宸渚?(Min MSE)] : {min_mse:.6e}")
    print(f"  [鏈€澶ц宸渚?(Max MSE)] : {max_mse:.6e}  (瀵瑰簲鏂囦欢: {worst_file})")
    print("==================================================")

    # 鑷姩灏嗚宸渶澶х殑鏂囦欢鐢诲浘灞曠ず鍑烘潵锛屼互渚挎帓鏌ユ槸鍝釜鐜妭鎷熷悎寰椾笉濂?
    if plot_worst_case and worst_file:
        print(f"\n>>> 姝ｅ湪缁樺埗璇樊鏈€澶х殑鏂囦欢 ({worst_file}) 浠ヤ緵鎺掓煡...")
        plt.figure(figsize=(12, 5))
        
        # S11 骞呭害
        plt.subplot(1, 2, 1)
        worst_nw_orig.plot_s_db(m=0, n=0, color='blue', label='HFSS $S_{11}$')
        worst_nw_pred.plot_s_db(m=0, n=0, color='red', linestyle='--', label='NN $S_{11}$')
        plt.title(f"{worst_file} - S11 Magnitude (dB)")
        plt.grid(True)
        
        # S21 骞呭害
        plt.subplot(1, 2, 2)
        worst_nw_orig.plot_s_db(m=1, n=0, color='blue', label='HFSS $S_{21}$')
        worst_nw_pred.plot_s_db(m=1, n=0, color='red', linestyle='--', label='NN $S_{21}$')
        plt.title(f"{worst_file} - S21 Magnitude (dB)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    for idx in range(300):
        # Calc_RDL_Top_S(idx)
        # Calc_RDL_Bottom_S(idx)
        # Calc_TSV_S(idx)
        Calc_Cascaded_RDL_TSV_S(idx)
        print(idx)
        # Batch_Calc_Cascaded_RDL_TSV_S()
    # Compare_Snp_Directories()
