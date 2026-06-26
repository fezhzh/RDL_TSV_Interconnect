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
# 1. 绁炵粡缃戠粶缁撴瀯瀹氫箟 (鍔犲叆鍔ㄦ€佽嚜閫傚簲濉厖)
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
        """褰撳簭鍒楅暱搴﹁涓嬮噰鏍峰埌灏忎簬鍗风Н鏍稿昂瀵告椂锛岃嚜鍔ㄥ湪鏈熬琛?0 闃叉宕╂簝"""
        if x.shape[-1] < kernel_size:
            pad_len = kernel_size - x.shape[-1]
            x = F.pad(x, (0, pad_len))
        return x

    def forward(self, x):
        original_length = x.shape[-1] # 璁板綍杈撳叆棰戠偣鏁?
        
        # 缂栫爜 (鑷姩妫€鏌ラ暱搴?+ ReLU 婵€娲?
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
        x = self.conv7(x) # 鐡堕灞備笉鍔犳縺娲?

        # 瑙ｇ爜
        x = F.relu(self.norm_dec8(self.tconv1(x)))
        x = F.relu(self.norm_dec9(self.tconv2(x)))
        x = F.relu(self.norm_dec10(self.tconv3(x)))
        x = F.relu(self.norm_dec11(self.tconv4(x)))
        x = self.tconv5(x) # 杈撳嚭灞?

        # 寮哄埗灏嗚緭鍑虹殑鐗瑰緛鍥惧昂瀵稿榻愬洖鍘熷棰戠偣鏁伴噺
        if x.shape[-1] != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
            
        return x

# ==========================================
# 2. 鏁版嵁澶勭悊涓庢彁鍙?
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
    print(f"姝ｅ湪浠?{input_dir} 鍜?{target_dir} 鏋勫缓鏁版嵁闆?..")
    
    matched_count = 0
    expected_length = None  # 鐢ㄤ簬璁板綍鏍囧噯棰戠偣鏁伴噺
    
    for file_in in input_files:
        filename = os.path.basename(file_in)
        file_target = os.path.join(target_dir, filename)
        
        if not os.path.exists(file_target):
            continue
            
        try:
            x_data = extract_4_channels_from_snp(file_in)
            y_data = extract_4_channels_from_snp(file_target)
            
            # 浠ヨ鍙栧埌鐨勭涓€缁勬垚鍔熺殑鏁版嵁闀垮害浣滀负鍩哄噯
            if expected_length is None:
                expected_length = x_data.shape[1]
                print(f">>> Set reference frequency point count: {expected_length}")
            
            # 濡傛灉褰撳墠鏂囦欢闀垮害涓庡熀鍑嗕笉涓€鑷达紝鐩存帴鎶涘純闃叉瀵艰嚧鎶ラ敊
            if x_data.shape[1] != expected_length or y_data.shape[1] != expected_length:
                print(f"[璺宠繃] {filename} 棰戠偣鏁颁笉涓€鑷? X: {x_data.shape[1]}, Y: {y_data.shape[1]}")
                continue
            
            X_list.append(x_data)
            Y_list.append(y_data)
            matched_count += 1
            
        except Exception as e:
            print(f"[鎶ラ敊] 澶勭悊 {filename} 鏃跺嚭閿欒烦杩? {e}")
            
    print(f"鏁版嵁闆嗘瀯寤哄畬鎴愶紒鎴愬姛閰嶅涓旂淮搴︿竴鑷寸殑鏍锋湰鏁? {matched_count}")
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
# 3. 璁粌杩囩▼鎺у埗
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
    print(f"浣跨敤璁惧: {device}")
    
    model = SNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    # 鍒涘缓涓撻棬瀛樻斁妯″瀷鐨勬枃浠跺す
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

        # ==== 鏂板锛氫繚瀛樻瘡涓€涓?epoch 鐨勬ā鍨?====
        epoch_model_path = os.path.join(ckpt_dir, f"SNN_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # ==== 淇濈暀鏈€浣虫ā鍨嬮€昏緫 ====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(ckpt_dir, "SNN_best.pth")
            torch.save(model.state_dict(), best_model_path)

    print(f"\n>>> 璁粌缁撴潫锛佹墍鏈夎凯浠ｆā鍨嬪凡淇濆瓨鑷?'{ckpt_dir}' 鏂囦欢澶癸紝鏈€浣虫ā鍨嬪凡淇濆瓨涓?'SNN_best.pth'")
    
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
# 5. 璁粌鍚庤瘎浼颁笌瀵规瘮缁樺浘
# ==========================================
def evaluate_and_plot(model_path, X_data, Y_data, num_plots=2):
    """
    鍔犺浇璁粌濂界殑妯″瀷锛屼粠鏁版嵁闆嗕腑闅忔満鎶藉彇鏍锋湰锛岀粯鍒?NN棰勬祴缁撴灉 涓?HFSS浠跨湡缁撴灉 鐨勫姣斿浘
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 瀹炰緥鍖栨ā鍨嬪苟鍔犺浇璁粌濂界殑鏉冮噸
    model = SNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"\n>>> 鎴愬姛鍔犺浇妯″瀷鏉冮噸: {model_path}")
    except Exception as e:
        print(f"鍔犺浇妯″瀷澶辫触: {e}")
        return

    # 2. 闅忔満鎶藉彇鍑犱釜鏍锋湰杩涜瀵规瘮
    num_samples = len(X_data)
    # 涓轰簡鐪嬪埌鐪熷疄鐨勬硾鍖栬兘鍔涳紝鎴戜滑灏介噺浠庨潬鍚庣殑绱㈠紩锛堥€氬父鏄鍒掑叆楠岃瘉闆嗙殑鍖哄煙锛夋娊鏍?
    # 纭繚鎶芥牱鏁伴噺涓嶄細澶т簬鍙敤鐨勬牱鏈暟
    actual_num_plots = min(num_plots, num_samples - int(num_samples * 0.8))
    if actual_num_plots <= 0: # 鏁版嵁閲忓お灏戠殑鎯呭喌瀹归敊
        actual_num_plots = min(num_plots, num_samples)
        sample_indices = np.random.choice(range(num_samples), actual_num_plots, replace=False)
    else:
        sample_indices = np.random.choice(range(int(num_samples * 0.8), num_samples), actual_num_plots, replace=False)

    for idx in sample_indices:
        # 鑾峰彇鍘熷鐗瑰緛鏁版嵁
        x_sample = X_data[idx:idx+1]  # 淇濇寔 batch 缁村害 [1, 4, L]
        y_truth  = Y_data[idx]        # 鐪熷疄鍊?[4, L]
        
        # 绁炵粡缃戠粶鍓嶅悜鎺ㄧ悊
        with torch.no_grad():
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).to(device)
            y_pred_tensor = model(x_tensor)
            y_pred = y_pred_tensor.cpu().numpy()[0] # [4, L]

        # 3. 鎻愬彇瀹為儴鍜岃櫄閮ㄥ苟璁＄畻 dB 骞呭害
        # 閫氶亾椤哄簭绾﹀畾涓? [S11_Real, S11_Imag, S21_Real, S21_Imag]
        def calc_db(real_part, imag_part):
            magnitude = np.sqrt(real_part**2 + imag_part**2)
            # 闃叉 log10(0) 鎶ラ敊锛屽姞涓婁竴涓瀬灏忓€?1e-12
            return 20 * np.log10(magnitude + 1e-12)

        # 璁＄畻鐪熷疄鍊?(HFSS) 鐨?dB
        s11_truth_db = calc_db(y_truth[0], y_truth[1])
        s21_truth_db = calc_db(y_truth[2], y_truth[3])

        # 璁＄畻棰勬祴鍊?(NN) 鐨?dB
        s11_pred_db  = calc_db(y_pred[0], y_pred[1])
        s21_pred_db  = calc_db(y_pred[2], y_pred[3])

        # 棰戠巼杞?(濡傛灉涓嶇煡閬撶‘鍒囬鐜囪寖鍥达紝鐩存帴鐢ㄧ偣鏁拌〃绀?
        freq_points = np.arange(len(s11_truth_db))

        # 4. 缁樺浘
        plt.figure(figsize=(12, 5))
        
        # --- 缁樺埗 S11 ---
        plt.subplot(1, 2, 1)
        plt.plot(freq_points, s11_truth_db, color='blue', linewidth=2, label='HFSS Simulated $S_{11}$')
        plt.plot(freq_points, s11_pred_db, color='red', linestyle='--', linewidth=2, label='NN Predicted $S_{11}$')
        plt.title(f"Sample #{idx} - S11 Magnitude")
        plt.xlabel("Frequency Points")
        plt.ylabel("Magnitude (dB)")
        plt.legend()
        plt.grid(True)
        
        # --- 缁樺埗 S21 ---
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
# 4. 涓荤▼搴?
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
        
    print(f"鏋勫缓鐨勬暟鎹淮搴? X={X_data.shape}, Y={Y_data.shape}")

    # 1. 鍚姩璁粌 (濡傛灉宸茬粡璁粌濂戒簡锛屾妸杩欒娉ㄩ噴鎺夊嵆鍙?
    # 鎺ユ敹杩斿洖鐨勬渶浣虫ā鍨嬭矾寰?
    final_model_path = train_mapping_network(X_data, Y_data, epochs=2000, batch_size=16, lr=0.001)

    # 2. 璁粌缁撴潫鍚庯紝鑷姩璇诲彇鏈€浣虫ā鍨嬪苟鐢诲浘瀵规瘮
    # 鍙傛暟 num_plots 淇敼涓?10锛岄殢鏈烘娊閫?10 涓牱鏈敾鍥?
    evaluate_and_plot(final_model_path, X_data, Y_data, num_plots=10)
