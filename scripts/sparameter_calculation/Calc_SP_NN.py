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
# 1. 鐗╃悊鍚彂鐨勮嚜寤烘ā鍧楋細CEL (鍥犳灉寰? & PEL (鏃犳簮鎬?
# ==========================================

class CEL(nn.Module):
    """
    鍥犳灉寰嬬害鏉熷眰 (Causality Enforcement Layer)
    鍩轰簬 PyTorch FFT 瀹炵幇 Kramers-Kronig 绾︽潫銆?
    灏?S 鍙傛暟杞叆鏃跺煙锛屾柦鍔犲洜鏋滅獥 (娑堥櫎 t<0 鐨勫搷搴?锛屽啀杞洖棰戝煙銆?
    """
    def __init__(self):
        super(CEL, self).__init__()

    def forward(self, x, freq_points):
        batch_size, channels, N = x.shape
        
        # 鎻愬彇澶嶆暟 S 鍙傛暟
        S11 = torch.complex(x[:, 0, :], x[:, 1, :])
        S21 = torch.complex(x[:, 2, :], x[:, 3, :])
        
        def enforce_causality_fft(S):
            # 1. 鏋勯€犲叡杞绉扮殑瀹屾暣棰戣氨浠ヨ繘琛屽倕閲屽彾閫嗗彉鎹?
            # 浼€犵洿娴?DC)鍒嗛噺锛屽彇绗竴涓鐐圭殑瀹為儴
            DC = torch.real(S[:, 0:1]).to(S.dtype)
            # 璐熼鐜囬儴鍒嗕负姝ｉ鐜囩殑鍏辫江鍊掑簭
            S_neg = torch.conj(torch.flip(S[:, :-1], dims=[1]))
            # 鎷兼帴: [DC, 姝ｉ鐜? 璐熼鐜嘳 (鎬婚暱搴︿负 2N)
            S_full = torch.cat([DC, S, S_neg], dim=1)
            
            # 2. IFFT 鍒版椂鍩?
            h_t = torch.fft.ifft(S_full, dim=1)
            
            # 3. 鏂藉姞鍥犳灉绐?(Causal Window): t=0澶勪负1, t>0澶勪负2, t<0澶勪负0
            window = torch.zeros_like(h_t, dtype=torch.float32)
            window[:, 0] = 1.0
            window[:, 1:N+1] = 2.0
            
            h_t_causal = h_t * window
            
            # 4. FFT 鍥炲埌棰戝煙
            S_causal_full = torch.fft.fft(h_t_causal, dim=1)
            
            # 5. 杩斿洖绾︽潫鍚庣殑姝ｉ鐜囬儴鍒?
            return S_causal_full[:, 1:N+1]

        S11_c = enforce_causality_fft(S11)
        S21_c = enforce_causality_fft(S21)
        
        # 灏嗗鏁版媶鍒嗕负瀹為儴鍜岃櫄閮?
        out = torch.zeros_like(x)
        out[:, 0, :] = torch.real(S11_c)
        out[:, 1, :] = torch.imag(S11_c)
        out[:, 2, :] = torch.real(S21_c)
        out[:, 3, :] = torch.imag(S21_c)
        return out


class PEL(nn.Module):
    """
    鏃犳簮鎬х害鏉熷眰 (Passivity Enforcement Layer)
    鍩轰簬 PyTorch SVD 濂囧紓鍊煎垎瑙ｅ疄鐜般€?
    淇濊瘉浠讳綍棰戠偣涓婄殑鏈€澶у寮傚€间笉瓒呰繃 1銆?
    """
    def __init__(self):
        super(PEL, self).__init__()

    def forward(self, x):
        batch_size, channels, freq_points = x.shape
        
        S11 = torch.complex(x[:, 0, :], x[:, 1, :])
        S21 = torch.complex(x[:, 2, :], x[:, 3, :])

        # 鏋勫缓浜掓槗缃戠粶鐨?2x2 鐭╅樀: [S11, S21 ; S21, S11]
        S_matrix = torch.stack([S11, S21, S21, S11], dim=1) 
        tensor = S_matrix.permute(0, 2, 1).reshape(batch_size, freq_points, 2, 2)
        
        # 濂囧紓鍊煎垎瑙?
        S_vals = torch.linalg.svdvals(tensor)
        max_val = S_vals.max(dim=-1).values
        
        # 濡傛灉濂囧紓鍊?> 1锛岃绠楃缉鏀炬瘮渚嬶紱鍚﹀垯淇濇寔涓?1
        scale = torch.where(max_val > 1.0, 1.0 / max_val, torch.ones_like(max_val))
        scale = scale.reshape(batch_size, 1, freq_points)

        # 鏂藉姞鏃犳簮缂╂斁
        S_pel = S_matrix * scale
        
        out = torch.zeros_like(x)
        out[:, 0, :] = torch.real(S_pel[:, 0, :])
        out[:, 1, :] = torch.imag(S_pel[:, 0, :])
        out[:, 2, :] = torch.real(S_pel[:, 1, :])
        out[:, 3, :] = torch.imag(S_pel[:, 1, :])
        return out


# ==========================================
# 2. 绁炵粡缃戠粶缁撴瀯瀹氫箟 (SNN)
# ==========================================
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv1d(4, 30, 32, 2, 0)
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

        # Decoder
        self.tconv1 = nn.ConvTranspose1d(30, 30, 32, 1, 0)
        self.norm_dec8 = nn.BatchNorm1d(30)
        self.tconv2 = nn.ConvTranspose1d(30, 30, 8, 2, 0)
        self.norm_dec9 = nn.BatchNorm1d(30)
        self.tconv3 = nn.ConvTranspose1d(30, 30, 4, 2, 0)
        self.norm_dec10 = nn.BatchNorm1d(30)
        self.tconv4 = nn.ConvTranspose1d(30, 30, 4, 4, 0)
        self.norm_dec11 = nn.BatchNorm1d(30)
        self.tconv5 = nn.ConvTranspose1d(30, 4, 4, 3, 0)

        # 瀹炰緥鍖栧師鐢熺害鏉熷眰
        self.CEL = CEL()
        self.PEL = PEL()

    def _pad_if_needed(self, x, kernel_size):
        if x.shape[-1] < kernel_size:
            pad_len = kernel_size - x.shape[-1]
            x = F.pad(x, (0, pad_len))
        return x

    def forward(self, x, current_iter=None, total_iter=None):
        original_length = x.shape[-1] 
        
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
        x = self.conv7(x) 

        x = F.relu(self.norm_dec8(self.tconv1(x)))
        x = F.relu(self.norm_dec9(self.tconv2(x)))
        x = F.relu(self.norm_dec10(self.tconv3(x)))
        x = F.relu(self.norm_dec11(self.tconv4(x)))
        x = self.tconv5(x) 

        if x.shape[-1] != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)

        # ====== 鏂藉姞鍐呭祵鐨勭墿鐞嗙害鏉?======
        # 1. 鍥犳灉寰?
        x = self.CEL(x, original_length)
        
        # 2. 鏃犳簮鎬?(璁粌鍚庢湡寮€鍚?
        if current_iter is not None and total_iter is not None:
            if current_iter >= (total_iter * 0.8): 
                x = self.PEL(x)
                
        return x

# ==========================================
# 3. 鏁版嵁闆嗘瀯寤轰笌娓呮礂
# ==========================================
def extract_4_channels_from_snp(filepath):
    nw = rf.Network(filepath)
    s11, s21 = nw.s[:, 0, 0], nw.s[:, 1, 0]
    return np.stack([s11.real, s11.imag, s21.real, s21.imag], axis=0).astype(np.float32)

def build_dataset(input_dir, target_dir):
    X_list, Y_list = [], []
    input_files = glob.glob(os.path.join(input_dir, "*.s*p"))
    print(f"\n>>> 姝ｅ湪浠?{input_dir} 鍜?{target_dir} 鏋勫缓鏁版嵁闆?..")
    
    matched_count = 0
    expected_length = None
    
    for file_in in input_files:
        filename = os.path.basename(file_in)
        file_target = os.path.join(target_dir, filename)
        
        if not os.path.exists(file_target): continue
            
        try:
            x_data = extract_4_channels_from_snp(file_in)
            y_data = extract_4_channels_from_snp(file_target)
            
            if expected_length is None:
                expected_length = x_data.shape[1]
                print(f"   [鍩哄噯闀垮害] 閿佸畾棰戠偣鏁? {expected_length}")
            
            if x_data.shape[1] != expected_length or y_data.shape[1] != expected_length:
                print(f"   [鍓旈櫎] 棰戠偣寮傚父: {filename}")
                continue
            
            X_list.append(x_data)
            Y_list.append(y_data)
            matched_count += 1
            
        except Exception as e:
            pass
            
    print(f">>> 鏋勫缓瀹屾瘯锛佹湁鏁堟牱鏈: {matched_count}\n")
    return np.array(X_list), np.array(Y_list)

class SParamDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = torch.tensor(X), torch.tensor(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 4. 璁粌鎺у埗
# ==========================================
def train_mapping_network(X_data, Y_data, epochs=150, batch_size=16, lr=0.001):
    num_samples = len(X_data)
    indices = np.random.permutation(num_samples)
    split = int(0.8 * num_samples)
    
    train_idx, val_idx = indices[:split], indices[split:]
    train_loader = DataLoader(SParamDataset(X_data[train_idx], Y_data[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SParamDataset(X_data[val_idx],   Y_data[val_idx]),   batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 寮€濮嬭缁冿紝褰撳墠纭欢: {device}")
    
    model = SNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # 鍓嶅悜浼犳挱锛屼紶鍏ュ綋鍓?epoch 鑷姩瑙﹀彂鍚庢湡鐨?PEL 绾︽潫
            pred_y = model(bx, current_iter=epoch, total_iter=epochs)
            loss = criterion(pred_y, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * bx.size(0)
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_hist.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred_y = model(bx, current_iter=epoch, total_iter=epochs)
                loss = criterion(pred_y, by)
                running_val_loss += loss.item() * bx.size(0)
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_hist.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.6e} | Val Loss: {avg_val_loss:.6e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "SNN_Cascade_to_HFSS.pth")

    print("\n>>> 璁粌瀹岀編鏀跺畼锛佹潈閲嶅凡淇濆瓨鑷?'SNN_Cascade_to_HFSS.pth'")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# 5. 璁粌鍚庣敾鍥惧姣?
# ==========================================
def evaluate_and_plot(model_path, X_data, Y_data, num_plots=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f">>> 鎴愬姛鍔犺浇鏉冮噸锛屽紑濮嬫娊鍙栭獙璇侀泦鏁版嵁鐢诲浘瀵规瘮...")
    except Exception as e:
        print(f"鍔犺浇妯″瀷澶辫触: {e}")
        return

    num_samples = len(X_data)
    sample_indices = np.random.choice(range(int(num_samples * 0.8), num_samples), num_plots, replace=False)

    for idx in sample_indices:
        x_sample = X_data[idx:idx+1]
        y_truth  = Y_data[idx]
        
        with torch.no_grad():
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).to(device)
            y_pred_tensor = model(x_tensor, current_iter=100, total_iter=100) # 瑙﹀彂 PEL
            y_pred = y_pred_tensor.cpu().numpy()[0]

        def calc_db(real_part, imag_part):
            return 20 * np.log10(np.sqrt(real_part**2 + imag_part**2) + 1e-12)

        s11_truth_db = calc_db(y_truth[0], y_truth[1])
        s21_truth_db = calc_db(y_truth[2], y_truth[3])
        s11_pred_db  = calc_db(y_pred[0], y_pred[1])
        s21_pred_db  = calc_db(y_pred[2], y_pred[3])
        s11_direct_db = calc_db(x_sample[0, 0], x_sample[0, 1])
        s21_direct_db = calc_db(x_sample[0, 2], x_sample[0, 3])
        freq_points = np.arange(len(s11_truth_db))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(freq_points, s11_truth_db, color='blue', linewidth=2, label='HFSS $S_{11}$')
        plt.plot(freq_points, s11_direct_db, color='green', linestyle=':', linewidth=1.5, label='Direct Cascade $S_{11}$')
        plt.plot(freq_points, s11_pred_db, color='red', linestyle='--', linewidth=2, label='NN $S_{11}$')
        plt.title(f"Sample #{idx} - S11 Magnitude")
        plt.xlabel("Frequency Points")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(freq_points, s21_truth_db, color='blue', linewidth=2, label='HFSS $S_{21}$')
        plt.plot(freq_points, s21_direct_db, color='green', linestyle=':', linewidth=1.5, label='Direct Cascade $S_{21}$')
        plt.plot(freq_points, s21_pred_db, color='red', linestyle='--', linewidth=2, label='NN $S_{21}$')
        plt.title(f"Sample #{idx} - S21 Magnitude")
        plt.xlabel("Frequency Points")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ==========================================
# 6. 缁熶竴鎵ц鍏ュ彛
# ==========================================
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])
    
    dir_input_X  = r"./snp_data/RDL_TSV_NN_Snp"    
    dir_target_Y = r"./snp_data/RDL_TSV_Snp"       

    if not os.path.exists(dir_input_X) or not os.path.exists(dir_target_Y):
        print("Error: data folder does not exist, please check paths.")
        exit()

    X_data, Y_data = build_dataset(dir_input_X, dir_target_Y)
    
    if len(X_data) < 10:
        print("Warning: too few valid data pairs were extracted; training may not converge.")
        exit()

    train_mapping_network(X_data, Y_data, epochs=1000, batch_size=16, lr=0.001)

    evaluate_and_plot("SNN_Cascade_to_HFSS.pth", X_data, Y_data, num_plots=2)
