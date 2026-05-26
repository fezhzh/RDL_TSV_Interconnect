import os
from pathlib import Path
import re
import numpy as np
import scipy.io as sio
import skrf as rf
import matplotlib.pyplot as plt
import glob
from scipy.optimize import least_squares

# ==========================================
# 1. 基础矩阵转换工具
# ==========================================
def s2abcd(S, Z0=50.0):
    """ 将 S 参数 (N, 2, 2) 转换为 ABCD 参数 (N, 2, 2) """
    S11 = S[:, 0, 0]
    S12 = S[:, 0, 1]
    S21 = S[:, 1, 0]
    S22 = S[:, 1, 1]
    
    denom = 2 * S21 + 1e-15 # 加极小值防止除以0
    
    A = ((1 + S11) * (1 - S22) + S12 * S21) / denom
    B = Z0 * ((1 + S11) * (1 + S22) - S12 * S21) / denom
    C_mat = (1 / Z0) * ((1 - S11) * (1 - S22) - S12 * S21) / denom
    D = ((1 - S11) * (1 + S22) + S12 * S21) / denom
    
    ABCD = np.zeros_like(S, dtype=complex)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C_mat
    ABCD[:, 1, 1] = D
    return ABCD

def abcd2s(ABCD, Z0=50.0):
    """ 将 ABCD 参数 (N, 2, 2) 转换为 S 参数 (N, 2, 2) """
    A = ABCD[:, 0, 0]
    B = ABCD[:, 0, 1]
    C_mat = ABCD[:, 1, 0]
    D = ABCD[:, 1, 1]
    
    denom = A + B/Z0 + C_mat*Z0 + D
    S11 = (A + B/Z0 - C_mat*Z0 - D) / denom
    S12 = 2 * (A*D - B*C_mat) / denom
    S21 = 2 / denom
    S22 = (-A + B/Z0 - C_mat*Z0 + D) / denom
    
    S = np.zeros_like(ABCD, dtype=complex)
    S[:, 0, 0] = S11
    S[:, 0, 1] = S12
    S[:, 1, 0] = S21
    S[:, 1, 1] = S22
    return S

# ==========================================
# 2. 从 s2p 文件中读取器件物理参数
# ==========================================
def extract_device_params_RDL_Top(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    try:
        length, width, thickness = params.get('lrdl'), params.get('wrdl'), params.get('trdl')
        htsv, p1 = params['htsv'], params['p1']
        return np.array([length, width, thickness, htsv, p1]), length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")

def extract_device_params_RDL_Bottom(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    try:
        length, width, thickness = params.get('ldown'), params.get('wdown'), params.get('tdown')
        htsv, p1 = params['htsv'], params['p1']
        return np.array([length, width, thickness, htsv, p1]), length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")

def extract_device_params_TSV(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    try:
        dtsv, length, p1 = params.get('dtsv'), params.get('htsv'), params['p1']
        return np.array([dtsv, length, p1]), length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")

def extract_device_params_RDL_TSV(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): break  
            if line.startswith('!'):
                line = line[1:].strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', val.strip())
                    if match:
                        params[key.strip()] = float(match.group(1))
    return params

# ==========================================
# 3. 神经网络推理
# ==========================================
def predict_circuit_parameters(features, mat_dir, param_names, prefix="RDL_Bottom_"):
    circuit_params = {}
    x = features.reshape(1, -1)
    for param in param_names:
        mat_filepath = os.path.join(mat_dir, f"{prefix}{param}.mat")
        if not os.path.exists(mat_filepath):
            circuit_params[param] = 1.0 
            continue
        mat_data = sio.loadmat(mat_filepath)
        xmin, xmax = mat_data['psmin'], mat_data['psmax']
        ymin, ymax = mat_data['outputmin'], mat_data['outputmax']
        w1, b1 = mat_data['w1'], mat_data['theta1']
        w2, b2 = mat_data['w2'], mat_data['theta2']
        w3, b3 = mat_data['w3'], mat_data['theta3']
        
        x_norm = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        a1 = np.tanh(np.dot(x_norm, w1) + b1)
        a2 = np.tanh(np.dot(a1, w2) + b2)
        y_norm = np.dot(a2, w3) + b3
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[param] = float(y_real.flatten()[0])
    return circuit_params

def calculate_S_parameters(circuit_params, length_um, freqs):
    R1, R2, R3 = circuit_params["R1"], circuit_params["R2"], circuit_params["R3"]
    L1, L2, L3 = circuit_params["L1"] * 1e-9, circuit_params["L2"] * 1e-9, circuit_params["L3"] * 1e-9
    Cox, Csi = circuit_params["Cox"] * 1e-12, circuit_params["Csi"] * 1e-12
    Rsi = circuit_params["Rsi"]
    
    length_m = length_um * 1e-6 
    omega = 2 * np.pi * freqs
    
    R_RLGC = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / ((R1 + R2)**2 + omega**2 * L2**2) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_RLGC = (R1**2 * L2) / ((R1 + R2)**2 + omega**2 * L2**2) + L3 * R3**2 / (R3**2 + omega**2 * L3**2) + L1
    G_RLGC = (omega**2 * Rsi * Cox**2) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)
    C_RLGC = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)

    Z0 = np.sqrt((R_RLGC + 1j * omega * L_RLGC) / (G_RLGC + 1j * omega * C_RLGC))
    GAMMA = np.sqrt((R_RLGC + 1j * omega * L_RLGC) * (G_RLGC + 1j * omega * C_RLGC))

    A = np.cosh(GAMMA * length_m)
    B = Z0 * np.sinh(GAMMA * length_m)
    C_mat = (1 / Z0) * np.sinh(GAMMA * length_m)
    D = np.cosh(GAMMA * length_m)
    
    denom = A + B/50.0 + C_mat*50.0 + D
    S11 = (A + B/50.0 - C_mat*50.0 - D) / denom
    S12 = 2 * (A*D - B*C_mat) / denom
    S21 = 2 / denom
    S22 = (-A + B/50.0 - C_mat*50.0 + D) / denom
    
    S_matrices = np.zeros((len(freqs), 2, 2), dtype=complex)
    S_matrices[:, 0, 0] = S11
    S_matrices[:, 0, 1] = S12
    S_matrices[:, 1, 0] = S21
    S_matrices[:, 1, 1] = S22
    return S_matrices

# ==========================================
# 4. 【修改】可视化对比 (加入直接级联网络)
# ==========================================
def Plot_S_Comparison(hfss_nw, nn_nw, direct_nw, title_suffix=""):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    hfss_nw.plot_s_db(m=0, n=0, color='blue', label='HFSS Simulated $S_{11}$')
    direct_nw.plot_s_db(m=0, n=0, color='green', linestyle=':', label='Direct Cascade $S_{11}$')
    nn_nw.plot_s_db(m=0, n=0, color='red', linestyle='--', label=f'Optimized $S_{11}$')
    plt.title(f"S11 Magnitude (dB) {title_suffix}")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    hfss_nw.plot_s_db(m=1, n=0, color='blue', label='HFSS Simulated $S_{21}$')
    direct_nw.plot_s_db(m=1, n=0, color='green', linestyle=':', label='Direct Cascade $S_{21}$')
    nn_nw.plot_s_db(m=1, n=0, color='red', linestyle='--', label=f'Optimized $S_{21}$')
    plt.title(f"S21 Magnitude (dB) {title_suffix}")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    mse_direct = np.mean(np.abs(hfss_nw.s - direct_nw.s)**2)
    mse_opt = np.mean(np.abs(hfss_nw.s - nn_nw.s)**2)
    print(f"\n>>> 直接级联 S 矩阵均方误差 (MSE): {mse_direct:.4e}")
    print(f">>> 优化修正 S 矩阵均方误差 (MSE): {mse_opt:.4e}")
    print(f">>> 误差降低了: {(mse_direct - mse_opt)/mse_direct * 100:.2f} %")

# ==========================================
# 5. ABCD 修正网络计算
# ==========================================
def get_correction_abcd(p, omega):
    """
    p: [Cn1_scale, Rn1_scale, Cn2_scale, Rn2_scale, Cn3_scale, Rn3_scale, Ln1_scale]
    利用缩放因子保证优化器在 0.1~1000 间正常工作，不报错
    """
    Cn1 = p[0] * 1e-14
    Rn1 = p[1] * 1e3
    Cn2 = p[2] * 1e-14
    Rn2 = p[3] * 1e3
    Cn3 = p[4] * 1e-14
    Rn3 = p[5] * 1.0
    Ln1 = p[6] * 1e-11

    Y1 = 1j * omega * Cn1 + 1.0 / Rn1
    Y2 = 1j * omega * Cn2 + 1.0 / Rn2
    Y3 = 1j * omega * Cn3 + 1.0 / (Rn3 + 1j * omega * Ln1)

    A = 1.0 + Y2 / Y3
    B = 1.0 / Y3
    C_val = Y1 + Y2 + Y1 * Y2 / Y3
    D = 1.0 + Y1 / Y3

    N = len(omega)
    ABCD = np.zeros((N, 2, 2), dtype=complex)
    ABCD[:, 0, 0] = A
    ABCD[:, 0, 1] = B
    ABCD[:, 1, 0] = C_val
    ABCD[:, 1, 1] = D
    return ABCD

# ==========================================
# 6. 级联计算与联合参数优化核心函数
# ==========================================
def Calc_Cascaded_RDL_TSV_S(idx):
    os.chdir(Path(__file__).resolve().parents[2]) 
    
    s2p_file = rf"./data/sparameters/RDL_TSV_Snp/dut{idx}.s2p"         
    mat_dir  = r"./data/matlab_models/RDL_TSV_mat2"                   
    
    if not os.path.exists(s2p_file):
        print(f"未找到测试文件: {s2p_file}")
        return

    print(f"\n>>> 开始处理 {s2p_file} 的参数优化...")
    Cascaded_HFSS_NW = rf.Network(s2p_file)
    freqs = Cascaded_HFSS_NW.f          
    omega = 2 * np.pi * freqs
    target_s = Cascaded_HFSS_NW.s
    
    # 提取物理参数
    params = extract_device_params_RDL_TSV(s2p_file)
    
    features_top = np.array([params['lrdl'], params['wrdl'], params['trdl'], params['htsv'], params['p1']])
    features_bot = np.array([params['ldown'], params['wdown'], params['tdown'], params['htsv'], params['p1']])
    features_tsv = np.array([params['dtsv'], params['htsv'], params['p1']])
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    # 预测基础的单体组件参数，并转换为 S 和 ABCD 参数
    cp_top = predict_circuit_parameters(features_top, mat_dir, target_params, prefix="RDL_Top_")
    s_top = calculate_S_parameters(cp_top, params['lrdl'], freqs)
    abcd_top = s2abcd(s_top)
    
    cp_bot = predict_circuit_parameters(features_bot, mat_dir, target_params, prefix="RDL_Bottom_")
    s_bot = calculate_S_parameters(cp_bot, params['ldown'], freqs)
    abcd_bot = s2abcd(s_bot)
    
    cp_tsv = predict_circuit_parameters(features_tsv, mat_dir, target_params, prefix="TSV_")
    s_tsv = calculate_S_parameters(cp_tsv, params['htsv'], freqs)
    abcd_tsv = s2abcd(s_tsv)
    
    # 定义 9 个模块的排列顺序
    base_abcds = [abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top, abcd_tsv, abcd_bot, abcd_tsv, abcd_top]

    # --- 【新增】计算纯直接级联的结果 ---
    direct_abcd = base_abcds[0]
    for i in range(1, 9):
        direct_abcd = np.matmul(direct_abcd, base_abcds[i])
    direct_s = abcd2s(direct_abcd)
    nw_cascade_direct = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=direct_s, name="Direct_Cascade")
    
    # --- 修正网络参数优化 ---
    def objective(p_all):
        res = base_abcds[0]
        for i in range(8):
            p_i = p_all[i*7 : (i+1)*7]
            corr_abcd = get_correction_abcd(p_i, omega)
            
            res = np.matmul(res, corr_abcd)
            res = np.matmul(res, base_abcds[i+1])
            
        pred_s = abcd2s(res)
        error = pred_s - target_s
        return np.concatenate([error.real.flatten(), error.imag.flatten()])

    # p0:Cn1; p1:Rn1; p2:Cn2; p3:Rn2; p4:Cn3; p5:Rn3; p6:Ln1
    p_init = [0.01, 1000.0, 0.01, 1000.0, 0, 0.01, 0.01] 
    p0 = np.tile(p_init, 8) 
    bounds = (-1e5, 1e5) 
    
    print(">>> 正在优化 8 个修正网络共 56 个拓扑参数...")
    res_opt = least_squares(objective, p0, bounds=bounds, max_nfev=300)
    
    print(">>> 优化完成！开始计算最终结果...")
    
    # 提取拟合参数计算优化后结果
    opt_abcd = base_abcds[0]
    for i in range(8):
        p_i = res_opt.x[i*7 : (i+1)*7]
        corr_abcd = get_correction_abcd(p_i, omega)
        opt_abcd = np.matmul(opt_abcd, corr_abcd)
        opt_abcd = np.matmul(opt_abcd, base_abcds[i+1])
        
    opt_s = abcd2s(opt_abcd)
    
    nw_cascade_opt = rf.Network(frequency=Cascaded_HFSS_NW.frequency, s=opt_s, name="Optimized_NN_Correction")
    Cascaded_HFSS_NW.name = "RDL_TSV_HFSS_Simulated"

    # 将 3 个 Network 传给画图函数
    Plot_S_Comparison(Cascaded_HFSS_NW, nw_cascade_opt, nw_cascade_direct, title_suffix=f"(dut{idx}.s2p)")

if __name__ == "__main__":
    for idx in range(1, 11):
        Calc_Cascaded_RDL_TSV_S(idx)