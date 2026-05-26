import os
from pathlib import Path
import re
import numpy as np
import scipy.io as sio
import skrf as rf
import matplotlib.pyplot as plt
import glob

# ==========================================
# 1. 从 s2p 文件中读取器件物理参数
# ==========================================
def extract_device_params_RDL_Top(filepath):
    """
    解析 .s2p 文件头部的注释，提取器件物理参数 (作为 NN 的输入 x)
    兼容 RDL_top (lrdl) 和 RDL_bottom (ldown)
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
                        
    # 动态适配 lrdl(Top) 或 ldown(Bottom)
    try:
        length = params.get('lrdl')
        width = params.get('wrdl')
        thickness = params.get('trdl')
        htsv = params['htsv']
        p1 = params['p1']
        
        features = np.array([length, width, thickness, htsv, p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")
    

def extract_device_params_RDL_Bottom(filepath):
    """
    解析 .s2p 文件头部的注释，提取器件物理参数 (作为 NN 的输入 x)
    兼容 RDL_top (lrdl) 和 RDL_bottom (ldown)
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
                        
    # 动态适配 lrdl(Top) 或 ldown(Bottom)
    try:
        length = params.get('ldown')
        width = params.get('wdown')
        thickness = params.get('tdown')
        htsv = params['htsv']
        p1 = params['p1']
        
        features = np.array([length, width, thickness, htsv, p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")

def extract_device_params_TSV(filepath):
    """
    解析 .s2p 文件头部的注释，提取器件物理参数 (作为 NN 的输入 x)
    兼容 RDL_top (lrdl) 和 RDL_bottom (ldown)
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
                        
    # 动态适配 lrdl(Top) 或 ldown(Bottom)
    try:
        dtsv = params.get('dtsv')
        length = params.get('htsv')
        p1 = params['p1']
        
        features = np.array([dtsv,length,p1])
        return features, length
    except KeyError as e:
        raise ValueError(f"S2P 文件缺少必要的参数注释: {e}")
    

def extract_device_params_RDL_TSV(filepath):
    """
    解析 ./RDL_TSV/dut{idx}.s2p 整体链路文件头部的注释。
    提取出包含 Top、Bottom 和 TSV 的所有物理参数。
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
# 2. 神经网络推理 (循环加载 9 个 .mat 文件)
# ==========================================
def predict_circuit_parameters(features, mat_dir, param_names, prefix="RDL_Bottom_"):
    """
    读取 MATLAB 导出的权值，预测出 9 个等效电路缩放参数 (nH, pF, Ohm)
    """
    circuit_params = {}
    x = features.reshape(1, -1)
    
    for param in param_names:
        mat_filepath = os.path.join(mat_dir, f"{prefix}{param}.mat")
        
        if not os.path.exists(mat_filepath):
            print(f"警告：未找到模型文件 {mat_filepath}，分配安全默认值。")
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
        
        # 归一化 [-1, 1]
        x_norm = 2.0 * (x - xmin) / (xmax - xmin + 1e-12) - 1.0
        
        # 前向传播 (双隐藏层 tansig -> purelin)
        a1 = np.tanh(np.dot(x_norm, w1) + b1)
        a2 = np.tanh(np.dot(a1, w2) + b2)
        y_norm = np.dot(a2, w3) + b3
        
        # 反归一化
        y_real = ymin + (y_norm + 1.0) * (ymax - ymin) / 2.0
        circuit_params[param] = float(y_real.flatten()[0])
        
    return circuit_params

# ==========================================
# 3. RLGC -> 计算 S 参数 (完美对齐您的频变数学模型)
# ==========================================
def calculate_S_parameters(circuit_params, length_um, freqs):
    """
    1. 反缩放网络输出的 nH 和 pF 参数到标准单位 (H, F)。
    2. 计算随频率变化的单位长度分布参数 (R_RLGC, L_RLGC, G_RLGC, C_RLGC)。
    3. 根据长度转化为 ABCD 矩阵并提取 S 矩阵。
    """
    # 提取并逆缩放网络参数：L 恢复为 H (乘 1e-9)，C 恢复为 F (乘 1e-12)
    R1 = circuit_params["R1"]
    R2 = circuit_params["R2"]
    R3 = circuit_params["R3"]
    L1 = circuit_params["L1"] * 1e-9
    L2 = circuit_params["L2"] * 1e-9
    L3 = circuit_params["L3"] * 1e-9
    Cox = circuit_params["Cox"] * 1e-12
    Csi = circuit_params["Csi"] * 1e-12
    Rsi = circuit_params["Rsi"]
    
    # 物理长度转换 (um -> m)
    length_m = length_um * 1e-6 
    omega = 2 * np.pi * freqs
    
    # === 使用 NumPy 向量化操作加速频变公式计算 (与提参2.py完全一致) ===
    # 串联阻抗支路 (趋肤与涡流)
    R_RLGC = (R1**2 * R2 + R1 * R2**2 + omega**2 * R1 * L2**2) / ((R1 + R2)**2 + omega**2 * L2**2) + (omega**2 * L3**2 * R3) / (R3**2 + omega**2 * L3**2)
    L_RLGC = (R1**2 * L2) / ((R1 + R2)**2 + omega**2 * L2**2) + L3 * R3**2 / (R3**2 + omega**2 * L3**2) + L1
    
    # 并联导纳支路 (硅衬底色散)
    G_RLGC = (omega**2 * Rsi * Cox**2) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)
    C_RLGC = (Cox + omega**2 * Csi * Rsi**2 * Cox * (Cox + Csi)) / (1 + omega**2 * Rsi**2 * (Cox + Csi)**2)

    # 计算特征阻抗与传播常数
    Z0 = np.sqrt((R_RLGC + 1j * omega * L_RLGC) / (G_RLGC + 1j * omega * C_RLGC))
    GAMMA = np.sqrt((R_RLGC + 1j * omega * L_RLGC) * (G_RLGC + 1j * omega * C_RLGC))

    # 传输线的 ABCD 参数
    A = np.cosh(GAMMA * length_m)
    B = Z0 * np.sinh(GAMMA * length_m)
    C_mat = (1 / Z0) * np.sinh(GAMMA * length_m)
    D = np.cosh(GAMMA * length_m)
    
    # ABCD 转 S 参数公式
    denom = A + B/50.0 + C_mat*50.0 + D
    S11 = (A + B/50.0 - C_mat*50.0 - D) / denom
    S12 = 2 * (A*D - B*C_mat) / denom
    S21 = 2 / denom
    S22 = (-A + B/50.0 - C_mat*50.0 + D) / denom
    
    # 组合为 S 矩阵组: (频点数, 2, 2)
    S_matrices = np.zeros((len(freqs), 2, 2), dtype=complex)
    S_matrices[:, 0, 0] = S11
    S_matrices[:, 0, 1] = S12
    S_matrices[:, 1, 0] = S21
    S_matrices[:, 1, 1] = S22
        
    return S_matrices

# ==========================================
# 4. 可视化对比
# ==========================================
def Plot_S_Comparison(hfss_nw, nn_nw):
    # 5. 可视化对比
    plt.figure(figsize=(12, 5))
    
    # 画 S11 幅度
    plt.subplot(1, 2, 1)
    hfss_nw.plot_s_db(m=0, n=0, color='blue', label='HFSS $S_{11}$')
    nn_nw.plot_s_db(m=0, n=0, color='red', linestyle='--', label='NN $S_{11}$')
    plt.title("S11 Magnitude (dB)")
    plt.grid(True)
    
    # 画 S21 幅度
    plt.subplot(1, 2, 2)
    hfss_nw.plot_s_db(m=1, n=0, color='blue', label='HFSS $S_{21}$')
    nn_nw.plot_s_db(m=1, n=0, color='red', linestyle='--', label='NN $S_{21}$')
    plt.title("S21 Magnitude (dB)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    mse = np.mean(np.abs(hfss_nw.s - nn_nw.s)**2)
    print(f"\n>>> 对比结束！整体 S 矩阵均方误差 (MSE): {mse:.4e}")


# ==========================================
# 5. 计算S参数并构造网络
# ==========================================

def Calc_RDL_Top_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 切换到当前脚本目录，确保路径正确
    s2p_file = rf"./data/sparameters/RDL_Top_Snp/dut{idx}.s2p"         # HFSS 原始测试数据
    mat_dir  = r"./data/matlab_models/RDL_TSV_mat2"                            # .mat 模型存放的目录
    model_prefix = "RDL_Top_"                # 神经网络导出的前缀
    
    if not os.path.exists(s2p_file):
        print(f"未找到测试文件: {s2p_file}")
        return

    # 1. 提取 HFSS 真实 S 参数
    print(">>> 正在加载 HFSS 原始 S参数文件...")
    RDL_Top_HFSS_NW = rf.Network(s2p_file)
    freqs = RDL_Top_HFSS_NW.f          
    
    # 2. 提取器件物理参数作为输入
    device_features, length_um = extract_device_params_RDL_Top(s2p_file)
    print(device_features)
    print(f">>> 提取到物理特征向量: {device_features}")
    
    # 3. 循环载入 9 个神经网络模型预测参数
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 正在循环推理神经网络，预测等效电路参数...")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 等效电路转化为预测的 S 矩阵
    print(">>> 正在恢复物理量级，并基于 RLGC 频变模型生成 S 参数...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    RDL_Top_NN_NW = rf.Network(frequency=RDL_Top_HFSS_NW.frequency, s=predicted_s_matrices, name="RDL_Top_NN_Predicted")
    RDL_Top_HFSS_NW.name = "RDL_Top_HFSS_Simulated"

    Plot_S_Comparison(RDL_Top_HFSS_NW, RDL_Top_NN_NW)

def Calc_RDL_Bottom_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 切换到当前脚本目录，确保路径正确
    s2p_file = rf"./data/sparameters/RDL_Bottom_Snp/dut{idx}.s2p"         # HFSS 原始测试数据
    mat_dir  = r"./data/matlab_models/RDL_TSV_mat2"                            # .mat 模型存放的目录
    model_prefix = "RDL_Bottom_"                # 神经网络导出的前缀
    
    if not os.path.exists(s2p_file):
        print(f"未找到测试文件: {s2p_file}")
        return

    # 1. 提取 HFSS 真实 S 参数
    print(">>> 正在加载 HFSS 原始 S参数文件...")
    RDL_Bottom_HFSS_NW = rf.Network(s2p_file)
    freqs = RDL_Bottom_HFSS_NW.f          
    
    # 2. 提取器件物理参数作为输入
    device_features, length_um = extract_device_params_RDL_Bottom(s2p_file)
    print(device_features)
    print(f">>> 提取到物理特征向量: {device_features}")
    
    # 3. 循环载入 9 个神经网络模型预测参数
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 正在循环推理神经网络，预测等效电路参数...")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 等效电路转化为预测的 S 矩阵
    print(">>> 正在恢复物理量级，并基于 RLGC 频变模型生成 S 参数...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    
    RDL_Bottom_NN_NW = rf.Network(frequency=RDL_Bottom_HFSS_NW.frequency, s=predicted_s_matrices, name="RDL_Bottom_NN_Predicted")
    RDL_Bottom_HFSS_NW.name = "RDL_Bottom_HFSS_Simulated"
    Plot_S_Comparison(RDL_Bottom_HFSS_NW, RDL_Bottom_NN_NW)

def Calc_TSV_S(idx):
    os.chdir(Path(__file__).resolve().parents[2])  # 切换到当前脚本目录，确保路径正确
    s2p_file = rf"./data/sparameters/TSV_Snp/dut{idx}.s2p"         # HFSS 原始测试数据
    mat_dir  = r"./data/matlab_models/RDL_TSV_mat2"                            # .mat 模型存放的目录
    model_prefix = "TSV_"                # 神经网络导出的前缀
    
    if not os.path.exists(s2p_file):
        print(f"未找到测试文件: {s2p_file}")
        return

    # 1. 提取 HFSS 真实 S 参数
    print(">>> 正在加载 HFSS 原始 S参数文件...")
    TSV_HFSS_NW = rf.Network(s2p_file)
    freqs = TSV_HFSS_NW.f          
    
    # 2. 提取器件物理参数作为输入
    device_features, length_um = extract_device_params_TSV(s2p_file)
    print(device_features)
    print(f">>> 提取到物理特征向量: {device_features}")
    
    # 3. 循环载入 9 个神经网络模型预测参数
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    print(">>> 正在循环推理神经网络，预测等效电路参数...")
    circuit_params = predict_circuit_parameters(device_features, mat_dir, target_params, prefix=model_prefix)
    
    for k, v in circuit_params.items():
        if "L" in k:
            print(f"    - {k} : {v:.4e} (nH)")
        elif "C" in k:
            print(f"    - {k} : {v:.4e} (pF)")
        else:
            print(f"    - {k} : {v:.4e} (Ohm)")
        
    # 4. 等效电路转化为预测的 S 矩阵
    print(">>> 正在恢复物理量级，并基于 RLGC 频变模型生成 S 参数...")
    predicted_s_matrices = calculate_S_parameters(circuit_params, length_um, freqs)
    TSV_NN_NW = rf.Network(frequency=TSV_HFSS_NW.frequency, s=predicted_s_matrices, name="TSV_NN_Predicted")
    TSV_HFSS_NW.name = "TSV_HFSS_Simulated"

    Plot_S_Comparison(TSV_HFSS_NW, TSV_NN_NW)

# ==========================================
# 5. 【新增】级联计算核心函数 
# ==========================================
def Calc_Cascaded_RDL_TSV_S(idx):
    os.chdir(Path(__file__).resolve().parents[2]) 
    
    # 指向长链路全级联的 HFSS 测试文件
    s2p_file = rf"./data/sparameters/RDL_TSV_Snp/dut{idx}.s2p"         
    mat_dir  = r"./data/matlab_models/RDL_TSV_mat2"                   
    
    if not os.path.exists(s2p_file):
        print(f"未找到测试文件: {s2p_file}")
        return

    # 1. 提取 HFSS 真实 S 参数
    print(">>> 正在加载 HFSS 原始全链路 S参数文件...")
    Cascaded_HFSS_NW = rf.Network(s2p_file)
    freqs = Cascaded_HFSS_NW.f          
    
    # 2. 从总体 s2p 文件中提取所有物理尺寸参数
    params = extract_device_params_RDL_TSV(s2p_file)
    print(f">>> 提取到物理参数: {params}")
    
    # 将提取出的参数分别打包给对应的 NN 预测模型
    features_top = np.array([params['lrdl'], params['wrdl'], params['trdl'], params['htsv'], params['p1']])
    features_bot = np.array([params['ldown'], params['wdown'], params['tdown'], params['htsv'], params['p1']])
    features_tsv = np.array([params['dtsv'], params['htsv'], params['p1']])
    
    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]
    
    # 3. 分别计算 RDL_Top, RDL_Bottom, TSV 的单体 S 参数
    print(">>> 正在预测单体组件的等效电路并计算 S 参数...")
    
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
    
    # 4. 执行级联矩阵运算 (skrf ** 操作符会自动将S转为T，相乘后转回S)
    print(">>> 正在执行 S 参数矩阵级联 (Top - TSV - Bot - TSV - Top - TSV - Bot - TSV - Top) ...")
    
    # 9 段级联计算
    nw_cascade = nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top
    
    nw_cascade.name = "9-Stage_Cascaded_NN_Predicted"
    Cascaded_HFSS_NW.name = "RDL_TSV_HFSS_Simulated"

    # 5. 可视化对比结果
    Plot_S_Comparison(Cascaded_HFSS_NW, nw_cascade)
    


def Batch_Calc_Cascaded_RDL_TSV_S():
    os.chdir(Path(__file__).resolve().parents[2]) 
    
    # 路径配置
    input_dir  = r"./data/sparameters/RDL_TSV_Snp"
    output_dir = r"./data/sparameters/RDL_TSV_NN_Snp"
    mat_dir    = r"./data/matlab_models/RDL_TSV_mat2"
    
    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f">>> 已创建输出目录: {output_dir}")

    # 获取输入目录下所有的 .s2p 文件
    s2p_files = glob.glob(os.path.join(input_dir, "*.s2p"))
    
    if not s2p_files:
        print(f"未在 {input_dir} 中找到任何 .s2p 测试文件！")
        return

    print(f">>> 共扫描到 {len(s2p_files)} 个 .s2p 文件，准备开始批量级联处理...\n")

    target_params = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"]

    # 遍历处理每一个 s2p 文件
    for s2p_file in s2p_files:
        filename = os.path.basename(s2p_file)  # 获取诸如 "dut0.s2p" 的纯文件名
        print(f"--- 正在处理: {filename} ---")
        
        # 1. 提取 HFSS 真实 S 参数（主要用于获取正确的频率点 frequency）
        try:
            Cascaded_HFSS_NW = rf.Network(s2p_file)
            freqs = Cascaded_HFSS_NW.f          
        except Exception as e:
            print(f"读取文件失败，跳过 {filename}。报错: {e}")
            continue
        
        # 2. 从总体 s2p 文件中提取所有物理尺寸参数
        try:
            params = extract_device_params_RDL_TSV(s2p_file)
        except Exception as e:
            print(f"提取物理参数失败，跳过 {filename}。报错: {e}")
            continue
            
        # 将提取出的参数分别打包给对应的 NN 预测模型
        features_top = np.array([params['lrdl'], params['wrdl'], params['trdl'], params['htsv'], params['p1']])
        features_bot = np.array([params['ldown'], params['wdown'], params['tdown'], params['htsv'], params['p1']])
        features_tsv = np.array([params['dtsv'], params['htsv'], params['p1']])
        
        # 3. 分别计算 RDL_Top, RDL_Bottom, TSV 的单体 S 参数
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
        
        # 4. 执行级联矩阵运算 
        # (Top - TSV - Bot - TSV - Top - TSV - Bot - TSV - Top)
        nw_cascade = nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top ** nw_tsv ** nw_bot ** nw_tsv ** nw_top
        
        # 设置生成的 Network 名称，防止导出的 touchstone 文件里有奇怪的注释名
        nw_cascade.name = filename.replace(".s2p", "")

        # 5. 保存级联得到的 S 参数到目标目录，名字保持一一对应
        nw_cascade.write_touchstone(filename=filename, dir=output_dir)
        print(f"    [成功] 已级联并保存至 -> {os.path.join(output_dir, filename)}\n")
        

def Compare_Snp_Directories(dir_original="./data/sparameters/RDL_TSV_Snp", dir_predicted="./data/sparameters/RDL_TSV_NN_Snp", plot_worst_case=True):
    """
    对比两个文件夹下同名的 S 参数文件。
    :param dir_original: 原始 HFSS S参数文件夹路径
    :param dir_predicted: 神经网络预测 S参数文件夹路径
    :param plot_worst_case: 是否在对比结束后自动画出误差最大的一组数据进行人工确认
    """
    os.chdir(Path(__file__).resolve().parents[2])
    
    if not os.path.exists(dir_original):
        print(f"错误: 找不到原始文件夹 {dir_original}")
        return
    if not os.path.exists(dir_predicted):
        print(f"错误: 找不到预测文件夹 {dir_predicted}")
        return

    # 获取原始文件夹中所有的 .s2p 文件
    s2p_files_orig = glob.glob(os.path.join(dir_original, "*.s*p"))
    
    if not s2p_files_orig:
        print(f"在 {dir_original} 中没有找到任何 S 参数文件。")
        return

    print("==================================================")
    print(f"开始批量对比 S 参数: ")
    print(f"  基准数据: {dir_original}")
    print(f"  待测数据: {dir_predicted}")
    print("==================================================")

    mse_list = []
    matched_count = 0
    worst_mse = -1
    worst_file = None

    # 遍历原始文件夹里的文件
    for orig_filepath in s2p_files_orig:
        filename = os.path.basename(orig_filepath)
        pred_filepath = os.path.join(dir_predicted, filename)
        
        # 检查预测文件夹中是否存在同名文件
        if not os.path.exists(pred_filepath):
            print(f"[跳过] 缺失对应文件: {filename}")
            continue
            
        try:
            nw_orig = rf.Network(orig_filepath)
            nw_pred = rf.Network(pred_filepath)
            
            # 计算该对文件的 MSE (均方误差)
            mse = np.mean(np.abs(nw_orig.s - nw_pred.s)**2)
            mse_list.append(mse)
            matched_count += 1
            
            # 记录误差最大的一组
            if mse > worst_mse:
                worst_mse = mse
                worst_file = filename
                worst_nw_orig = nw_orig
                worst_nw_pred = nw_pred
                
        except Exception as e:
            print(f"[报错] 处理文件 {filename} 时发生异常: {e}")

    # ================= 统计与报告 =================
    if matched_count == 0:
        print("没有找到任何能成对匹配的文件！")
        return

    mse_array = np.array(mse_list)
    avg_mse = np.mean(mse_array)
    max_mse = np.max(mse_array)
    min_mse = np.min(mse_array)

    print(f"\n对比完成！成功配对文件数: {matched_count}")
    print(f"----------------------------------------")
    print(f"  [整体平均误差 (Avg MSE)] : {avg_mse:.6e}")
    print(f"  [最小误差案例 (Min MSE)] : {min_mse:.6e}")
    print(f"  [最大误差案例 (Max MSE)] : {max_mse:.6e}  (对应文件: {worst_file})")
    print("==================================================")

    # 自动将误差最大的文件画图展示出来，以便排查是哪个环节拟合得不好
    if plot_worst_case and worst_file:
        print(f"\n>>> 正在绘制误差最大的文件 ({worst_file}) 以供排查...")
        plt.figure(figsize=(12, 5))
        
        # S11 幅度
        plt.subplot(1, 2, 1)
        worst_nw_orig.plot_s_db(m=0, n=0, color='blue', label='HFSS $S_{11}$')
        worst_nw_pred.plot_s_db(m=0, n=0, color='red', linestyle='--', label='NN $S_{11}$')
        plt.title(f"{worst_file} - S11 Magnitude (dB)")
        plt.grid(True)
        
        # S21 幅度
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