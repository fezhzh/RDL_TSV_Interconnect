import os
from pathlib import Path
import glob
import numpy as np
import skrf as rf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. 神经网络结构定义 (加入动态自适应填充)
# ==========================================
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        #Fully Connected Layers
        self.conv1 = nn.Conv1d(4, 30,32,2,0)
        self.norm_dec1 = nn.BatchNorm1d(30)

        self.conv2 = nn.Conv1d(30, 30, 5, 1, 0)
        self.norm_dec2 = nn.BatchNorm1d(30)

        self.conv3 = nn.Conv1d(30, 30, 4, 2, 0)
        self.norm_dec3 = nn.BatchNorm1d(30)

        self.conv4 = nn.Conv1d(30, 30, 3, 1, 0)
        self.norm_dec4 = nn.BatchNorm1d(30)

        self.conv5 = nn.Conv1d(30, 30, 4, 4, 0)
        self.norm_dec5 = nn.BatchNorm1d(30)

        self.conv6 = nn.Conv1d(30, 30, 4, 2, 0)
        self.norm_dec6 = nn.BatchNorm1d(30)

        self.conv7 = nn.Conv1d(30, 30, 4, 2, 0)

        #Transposed Convolution Layers
        self.tconv1 = nn.ConvTranspose1d(30, 30, 32, 1, 0)
        self.norm_dec8 = nn.BatchNorm1d(30)

        self.tconv2 = nn.ConvTranspose1d(30, 30, 8, 2, 0)
        self.norm_dec9 = nn.BatchNorm1d(30)

        self.tconv3 = nn.ConvTranspose1d(30, 30, 4, 2, 0)
        self.norm_dec10 = nn.BatchNorm1d(30)

        self.tconv4 = nn.ConvTranspose1d(30, 30, 4, 4, 0)
        self.norm_dec11 = nn.BatchNorm1d(30)

        self.tconv5 = nn.ConvTranspose1d(30, 4, 4, 3, 0)

    def _pad_if_needed(self, x, kernel_size):
        """当序列长度被下采样到小于卷积核尺寸时，自动在末尾补 0 防止崩溃"""
        if x.shape[-1] < kernel_size:
            pad_len = kernel_size - x.shape[-1]
            x = F.pad(x, (0, pad_len))
        return x

    def forward(self, x):
        original_length = x.shape[-1] # 记录输入频点数
        
        # 编码 (自动检查长度 + ReLU 激活)
        x = self._pad_if_needed(x, self.conv1.kernel_size[0])
        x = F.relu(self.norm_dec1(self.conv1(x)))
        
        x = self._pad_if_needed(x, self.conv2.kernel_size[0])
        x = F.relu(self.norm_dec2(self.conv2(x)))
        
        x = self._pad_if_needed(x, self.conv3.kernel_size[0])
        x = F.relu(self.norm_dec3(self.conv3(x)))
        
        x = self._pad_if_needed(x, self.conv4.kernel_size[0])
        x = F.relu(self.norm_dec4(self.conv4(x)))
        
        x = self._pad_if_needed(x, self.conv5.kernel_size[0])
        x = F.relu(self.norm_dec5(self.conv5(x)))
        
        x = self._pad_if_needed(x, self.conv6.kernel_size[0])
        x = F.relu(self.norm_dec6(self.conv6(x)))
        
        x = self._pad_if_needed(x, self.conv7.kernel_size[0])
        x = self.conv7(x) # 瓶颈层不加激活

        # 解码
        x = F.relu(self.norm_dec8(self.tconv1(x)))
        x = F.relu(self.norm_dec9(self.tconv2(x)))
        x = F.relu(self.norm_dec10(self.tconv3(x)))
        x = F.relu(self.norm_dec11(self.tconv4(x)))
        x = self.tconv5(x) # 输出层

        # 强制将输出的特征图尺寸对齐回原始频点数量
        if x.shape[-1] != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
            
        return x

# ==========================================
# 2. 数据处理与提取
# ==========================================
def extract_4_channels_from_snp(filepath):
    nw = rf.Network(filepath)
    s11 = nw.s[:, 0, 0]
    s21 = nw.s[:, 1, 0]
    features = np.stack([s11.real, s11.imag, s21.real, s21.imag], axis=0)
    return features.astype(np.float32)

def build_dataset(input_dir, target_dir):
    X_list = []
    Y_list = []
    input_files = glob.glob(os.path.join(input_dir, "*.s*p"))
    print(f"正在从 {input_dir} 和 {target_dir} 构建数据集...")
    
    matched_count = 0
    expected_length = None  # 用于记录标准频点数量
    
    for file_in in input_files:
        filename = os.path.basename(file_in)
        file_target = os.path.join(target_dir, filename)
        
        if not os.path.exists(file_target):
            continue
            
        try:
            x_data = extract_4_channels_from_snp(file_in)
            y_data = extract_4_channels_from_snp(file_target)
            
            # 以读取到的第一组成功的数据长度作为基准
            if expected_length is None:
                expected_length = x_data.shape[1]
                print(f">>> 设定基准频点数量为: {expected_length} 点")
            
            # 如果当前文件长度与基准不一致，直接抛弃防止导致报错
            if x_data.shape[1] != expected_length or y_data.shape[1] != expected_length:
                print(f"[跳过] {filename} 频点数不一致! X: {x_data.shape[1]}, Y: {y_data.shape[1]}")
                continue
            
            X_list.append(x_data)
            Y_list.append(y_data)
            matched_count += 1
            
        except Exception as e:
            print(f"[报错] 处理 {filename} 时出错跳过: {e}")
            
    print(f"数据集构建完成！成功配对且维度一致的样本数: {matched_count}")
    return np.array(X_list), np.array(Y_list)

class SParamDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. 训练过程控制
# ==========================================
def train_mapping_network(X_data, Y_data, epochs=150, batch_size=16, lr=0.001):
    num_samples = len(X_data)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = SParamDataset(X_data[train_idx], Y_data[train_idx])
    val_dataset = SParamDataset(X_data[val_idx], Y_data[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = SNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    # 创建专门存放模型的文件夹
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            pred_y = model(bx)
            loss = criterion(pred_y, by)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * bx.size(0)
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred_y = model(bx)
                loss = criterion(pred_y, by)
                running_val_loss += loss.item() * bx.size(0)
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.6e} | Val Loss: {avg_val_loss:.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ==== 新增：保存每一个 epoch 的模型 ====
        epoch_model_path = os.path.join(ckpt_dir, f"SNN_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # ==== 保留最佳模型逻辑 ====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(ckpt_dir, "SNN_best.pth")
            torch.save(model.state_dict(), best_model_path)

    print(f"\n>>> 训练结束！所有迭代模型已保存至 '{ckpt_dir}' 文件夹，最佳模型已保存为 'SNN_best.pth'")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_model_path

# ==========================================
# 5. 训练后评估与对比绘图
# ==========================================
def evaluate_and_plot(model_path, X_data, Y_data, num_plots=2):
    """
    加载训练好的模型，从数据集中随机抽取样本，绘制 NN预测结果 与 HFSS仿真结果 的对比图
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型并加载训练好的权重
    model = SNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"\n>>> 成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 随机抽取几个样本进行对比
    num_samples = len(X_data)
    # 为了看到真实的泛化能力，我们尽量从靠后的索引（通常是被划入验证集的区域）抽样
    # 确保抽样数量不会大于可用的样本数
    actual_num_plots = min(num_plots, num_samples - int(num_samples * 0.8))
    if actual_num_plots <= 0: # 数据量太少的情况容错
        actual_num_plots = min(num_plots, num_samples)
        sample_indices = np.random.choice(range(num_samples), actual_num_plots, replace=False)
    else:
        sample_indices = np.random.choice(range(int(num_samples * 0.8), num_samples), actual_num_plots, replace=False)

    for idx in sample_indices:
        # 获取原始特征数据
        x_sample = X_data[idx:idx+1]  # 保持 batch 维度 [1, 4, L]
        y_truth  = Y_data[idx]        # 真实值 [4, L]
        
        # 神经网络前向推理
        with torch.no_grad():
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).to(device)
            y_pred_tensor = model(x_tensor)
            y_pred = y_pred_tensor.cpu().numpy()[0] # [4, L]

        # 3. 提取实部和虚部并计算 dB 幅度
        # 通道顺序约定为: [S11_Real, S11_Imag, S21_Real, S21_Imag]
        def calc_db(real_part, imag_part):
            magnitude = np.sqrt(real_part**2 + imag_part**2)
            # 防止 log10(0) 报错，加上一个极小值 1e-12
            return 20 * np.log10(magnitude + 1e-12)

        # 计算真实值 (HFSS) 的 dB
        s11_truth_db = calc_db(y_truth[0], y_truth[1])
        s21_truth_db = calc_db(y_truth[2], y_truth[3])

        # 计算预测值 (NN) 的 dB
        s11_pred_db  = calc_db(y_pred[0], y_pred[1])
        s21_pred_db  = calc_db(y_pred[2], y_pred[3])

        # 频率轴 (如果不知道确切频率范围，直接用点数表示)
        freq_points = np.arange(len(s11_truth_db))

        # 4. 绘图
        plt.figure(figsize=(12, 5))
        
        # --- 绘制 S11 ---
        plt.subplot(1, 2, 1)
        plt.plot(freq_points, s11_truth_db, color='blue', linewidth=2, label='HFSS Simulated $S_{11}$')
        plt.plot(freq_points, s11_pred_db, color='red', linestyle='--', linewidth=2, label='NN Predicted $S_{11}$')
        plt.title(f"Sample #{idx} - S11 Magnitude")
        plt.xlabel("Frequency Points")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        
        # --- 绘制 S21 ---
        plt.subplot(1, 2, 2)
        plt.plot(freq_points, s21_truth_db, color='blue', linewidth=2, label='HFSS Simulated $S_{21}$')
        plt.plot(freq_points, s21_pred_db, color='red', linestyle='--', linewidth=2, label='NN Predicted $S_{21}$')
        plt.title(f"Sample #{idx} - S21 Magnitude")
        plt.xlabel("Frequency Points")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    
    dir_input_X  = r"./data/sparameters/RDL_TSV_NN_Snp"    
    dir_target_Y = r"./data/sparameters/RDL_TSV_Snp"        

    if not os.path.exists(dir_input_X) or not os.path.exists(dir_target_Y):
        print("错误：数据文件夹不存在，请检查路径。")
        exit()

    X_data, Y_data = build_dataset(dir_input_X, dir_target_Y)
    
    if len(X_data) < 10:
        print("警告：提取到的有效数据对太少，网络难以收敛。")
        
    print(f"构建的数据维度: X={X_data.shape}, Y={Y_data.shape}")

    # 1. 启动训练 (如果已经训练好了，把这行注释掉即可)
    # 接收返回的最佳模型路径
    final_model_path = train_mapping_network(X_data, Y_data, epochs=2000, batch_size=16, lr=0.001)

    # 2. 训练结束后，自动读取最佳模型并画图对比
    # 参数 num_plots 修改为 10，随机抽选 10 个样本画图
    evaluate_and_plot(final_model_path, X_data, Y_data, num_plots=10)