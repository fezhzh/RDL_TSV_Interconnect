import os
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
from coordConv import addCoords_1D
import numpy as np
import matplotlib.pyplot as plt
import torch
from CEL import CEL
import torch.fft as fft
import pickle

with open("testx4.pkl", "rb") as f:
    loaded_x4 = pickle.load(f)
with open("testy4.pkl", "rb") as f:
    loaded_y4 = pickle.load(f)

def PEL_test(output2):
    num_samples = output2.shape[0]
    S_cel = torch.cat(
        [torch.complex(output2[:, 0, :], output2[:, 1, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]),torch.complex(output2[:, 0, :], output2[:, 1, :])],
        dim=1)
    S_cel=S_cel.view(num_samples, 4, 1000)
    batch_size, channels, orl = S_cel.shape
    tensor = S_cel.permute(0, 2, 1).reshape(batch_size, orl, 2, 2)
    S = torch.linalg.svdvals(tensor)
    value = S.max(dim=-1).values

    value = torch.where(value > 1, 1/ value, torch.ones_like(value))
    value = value.reshape(batch_size, 1, orl)
    # angle = torch.log(value)
    # angle = torch.cat([angle, torch.flip(angle[:, :, 0:-1], [2])], dim=2)
    # angle = torch.fft.fft(angle, dim=2)
    # angle = torch.cat([angle[:, :, 0].unsqueeze(1), 2.0 * angle[:, :, 1:orl],
    #                    0.0 * 2.0 * angle[:, :, orl:]], dim=2)
    # angle = torch.fft.ifft(angle, dim=2)
    # angle = angle[:, :, :orl]
    # passivity_filter = value * torch.exp(1j * angle)
    S_pel =  S_cel* value
    # S_pel =  S_cel * passivity_filter
    value.reshape(batch_size * orl)
    output2[:, 0, :] = torch.real(S_pel[:, 0, :])
    output2[:, 1, :] = torch.imag(S_pel[:, 0, :])
    output2[:, 2, :] = torch.real(S_pel[:, 1, :])
    output2[:, 3, :] = torch.imag(S_pel[:, 1, :])
    return output2

class PEL(nn.Module):
    def forward(self,output2,batch_size=32):
        num_samples = output2.shape[0]
        output3=torch.zeros((num_samples, 4, 1000))
        S_cel = torch.cat(
            [torch.complex(output2[:, 0, :], output2[:, 1, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]),
             torch.complex(output2[:, 2, :], output2[:, 3, :]), torch.complex(output2[:, 0, :], output2[:, 1, :])],
            dim=1)
        S_cel = S_cel.view(num_samples, 4, 1000)
        batch_size, channels, orl = S_cel.shape
        tensor = S_cel.permute(0, 2, 1).reshape(batch_size, orl, 2, 2)
        S = torch.linalg.svdvals(tensor)
        value = S.max(dim=-1).values
        value = torch.where(value > 1, 1 / value, torch.ones_like(value))
        value = value.reshape(batch_size, 1, orl)

        # angle = torch.log(value)
        # angle = torch.cat([angle, torch.flip(angle[:, :, 0:-1], [2])], dim=2)
        # angle = torch.fft.fft(angle, dim=2)
        # angle = torch.cat([angle[:, :, 0].unsqueeze(1), 2.0 * angle[:, :, 1:orl],
        #                    0.0 * 2.0 * angle[:, :, orl:]], dim=2)
        # angle = torch.fft.ifft(angle, dim=2)
        # angle = angle[:, :, :orl]
        # passivity_filter = value * torch.exp(1j * angle)

        S_pel = S_cel * value
        # S_pel = S_cel * passivity_filter
        value.reshape(batch_size * orl)
        output3[:, 0, :] = torch.real(S_pel[:, 0, :])
        output3[:, 1, :] = torch.imag(S_pel[:, 0, :])
        output3[:, 2, :] = torch.real(S_pel[:, 1, :])
        output3[:, 3, :] = torch.imag(S_pel[:, 1, :])
        return output3

class Solenoid_STCNN_V2(nn.Module):
    def __init__(self):
        super(Solenoid_STCNN_V2, self).__init__()
        #Fully Connected Layers
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(5, 50)
        self.lin2 = nn.Linear(50, 50)
        self.lin3 = nn.Linear(50, 50)
        self.lin4 = nn.Linear(50, 50)

        #Transposed Convolution Layers
        self.tconv1 = nn.ConvTranspose1d(50, 30, 32, 2, 0)
        self.norm_dec1 = nn.BatchNorm1d(30)

        self.tconv2 = nn.ConvTranspose1d(30, 30, 8, 2, 0)
        self.norm_dec2 = nn.BatchNorm1d(30)

        self.tconv3 = nn.ConvTranspose1d(30, 30, 4, 4, 0)
        self.norm_dec3 = nn.BatchNorm1d(30)

        self.tconv4 = nn.ConvTranspose1d(30, 30, 4, 2, 0)
        self.norm_dec4 = nn.BatchNorm1d(30)

        self.tconv5 = nn.ConvTranspose1d(30, 4, 4, 3, 0)
        # self.tconv5 = nn.ConvTranspose1d(25, 2, 4, 1, 0)
        self.CEL = CEL()
        self.PEL = PEL()

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = F.elu(latent)

        latent = self.lin2(latent)
        latent = F.elu(latent)

        latent = self.lin3(latent)
        latent = F.elu(latent)

        latent = self.lin4(latent)
        latent = F.elu(latent)
        z = latent.view(-1, self.lin4.out_features, 1)
        return z

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = F.elu(latent)

        latent = self.tconv2(latent)
        latent = F.elu(latent)

        latent = self.tconv3(latent)
        latent = F.elu(latent)

        latent = self.tconv4(latent)
        latent = F.elu(latent)

        recons_y = self.tconv5(latent)
        return recons_y[..., :1500]

    def forward(self, x, iter=None, total_iter=None):
        z = self.fully_connected(x)
        out1 = self.transposed_conv(z)
        out2=self.CEL(out1,1500)
        if iter is not None and total_iter is not None and iter >= (total_iter - 500):
            out2 = self.PEL(out2)
        # out2 = torch.clamp(out2, min=-1, max=1)
        return out2


cwd = os.getcwd()

#Device to be used in training. If GPU is available, it will be automatically used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# absolute_path = 'C:/Users/45042/PycharmProjects/pythonProject1/tsv-rdlinput.mat'
# # 使用绝对路径加载数据
# dataset = scipy.io.loadmat(absolute_path)
# inputrdl= dataset['trainingInputsrdl']
# inputtsv= dataset['trainingInputstsv']
# model1 = Solenoid_STCNN_V2().to(device)
# # 载入模型权重，假设cwd变量是包含模型文件的目录的字符串
# save_name_last_iter1 = "rdlup_last_iter.pth"
# model1.load_state_dict(torch.load(cwd + "/" + save_name_last_iter1))
# input_x = inputrdl
# meanX = input_x.min(axis=0)
# stdX = input_x.max(axis=0) - input_x.min(axis=0)
# input_x_normalized = -1+2*(input_x-meanX)/stdX
# input_x = torch.tensor(input_x_normalized)
# input_x = input_x.float()
# model1.eval()
# tensor_up=model1(input_x)
# model2 = Solenoid_STCNN_V2().to(device)
# # 载入模型权重，假设cwd变量是包含模型文件的目录的字符串
# save_name_last_iter1 = "rdldown_last_iter.pth"
# model2.load_state_dict(torch.load(cwd + "/" + save_name_last_iter1))
# input_x = inputrdl
# meanX = input_x.min(axis=0)
# stdX = input_x.max(axis=0) - input_x.min(axis=0)
# input_x_normalized = -1+2*(input_x-meanX)/stdX
# input_x = torch.tensor(input_x_normalized)
# input_x = input_x.float()
# model2.eval()
# tensor_down=model2(input_x)
#
# model3 = Solenoid_STCNN_V2().to(device)
# # 载入模型权重，假设cwd变量是包含模型文件的目录的字符串
# save_name_last_iter1 = "tsv_last_iter.pth"
# model3.load_state_dict(torch.load(cwd + "/" + save_name_last_iter1))
# input_x = inputtsv
# meanX = input_x.min(axis=0)
# stdX = input_x.max(axis=0) - input_x.min(axis=0)
# input_x_normalized = -1+2*(input_x-meanX)/stdX
# input_x = torch.tensor(input_x_normalized)
# input_x = input_x.float()
# model3.eval()
# tensor_tsv=model3(input_x)
# torch.save(tensor_down, 'radldown.pt')
# torch.save(tensor_up, 'radlup.pt')
# torch.save(tensor_tsv, 'tsv.pt')

def S_parameters_to_ABCD(output1, m, n):
    A = torch.zeros(m, n, dtype=torch.cfloat)
    B = torch.zeros(m, n, dtype=torch.cfloat)
    C = torch.zeros(m, n, dtype=torch.cfloat)
    D = torch.zeros(m, n, dtype=torch.cfloat)

    complex_tensor_1 = torch.complex(output1[:, 0, :], output1[:, 1, :])
    complex_tensor_2 = torch.complex(output1[:, 2, :], output1[:, 3, :])
    S = torch.stack([complex_tensor_1, complex_tensor_2], dim=1)
    S11 = S[:, 0, :]
    S21 = S[:, 1, :]

    for i in range(m):
        for j in range(n):
            A[i, j] = ((1 + S11[i, j]) * (1 - S11[i, j]) + S21[i, j] * S21[i, j]) / (2 * S21[i, j])
            B[i, j] = 50 * ((1 + S11[i, j]) * (1 + S11[i, j]) - S21[i, j] * S21[i, j]) / (2 * S21[i, j])
            C[i, j] = 1 / 50 * ((1 - S11[i, j]) * (1 - S11[i, j]) - S21[i, j] * S21[i, j]) / (2 * S21[i, j])
            D[i, j] = ((1 - S11[i, j]) * (1 + S11[i, j]) + S21[i, j] * S21[i, j]) / (2 * S21[i, j])

    ABCD = torch.zeros((m, 2, 2, n), dtype=torch.cfloat)

    # Fill the three-dimensional array with the elements from A, B, C, and D
    for i in range(m):
        for j in range(n):
            ABCD[i, 0, 0, j] = A[i, j]
            ABCD[i, 0, 1, j] = B[i, j]
            ABCD[i, 1, 0, j] = C[i, j]
            ABCD[i, 1, 1, j] = D[i, j]

    return ABCD

def ABCD_to_S_parameter(ABCD,m,n):
    s11 =  torch.zeros(m, n, dtype=torch.cfloat)
    s12 =  torch.zeros(m, n, dtype=torch.cfloat)

    for i in range(m):
        for j in range(n):
            s11[i,j] = (ABCD[i, 0, 0,j] + ABCD[i, 0, 1,j] / 50 - ABCD[i, 1, 0,j] * 50 - ABCD[i, 1, 1,j]) / (
                    ABCD[i, 0, 0,j] + ABCD[i, 0, 1,j] / 50 + ABCD[i, 1, 0,j] * 50 + ABCD[i, 1, 1,j])

            s12[i,j] = 2 * (ABCD[i, 0, 0,j] * ABCD[i, 1, 1,j] - ABCD[i, 0, 1,j] * ABCD[i, 1, 0,j]) / (
                    ABCD[i, 0, 0,j] + ABCD[i, 0, 1,j] / 50 + ABCD[i, 1, 0,j] * 50 + ABCD[i, 1, 1,j])
    s11_real=torch.real(s11)
    s11_imag=torch.imag(s11)
    s12_real = torch.real(s12)
    s12_imag = torch.imag(s12)
    S = torch.stack([s11_real, s11_imag, s12_real, s12_imag], dim=1)
    return S

def ABCD_cascading(ABCD1,ABCD2,ABCD3,m,n):
    ABCD = torch.zeros((m, 2, 2, n), dtype=torch.cfloat)
    for i in range(m):
        for j in range(n):
            ABCD[i, :, :, j] = ABCD1[i, :, :, j] @ ABCD2[i, :, :, j] @ ABCD3[i, :, :, j]
    return ABCD

tensor_up=torch.load('radlup.pt')
tensor_tsv=torch.load('tsv.pt')
tensor_down=torch.load('radldown.pt')
tensor_corner=torch.load('corner.pt')
m=200
n=1000
# ABCD1 = S_parameters_to_ABCD(tensor_up, m, n)
# S1 = ABCD_to_S_parameter(ABCD1,m, n)
# ABCD2 = S_parameters_to_ABCD(tensor_tsv, m, n)
# S2 = ABCD_to_S_parameter(ABCD1,m, n)
# ABCD3 = S_parameters_to_ABCD(tensor_down, m, n)
# S3 = ABCD_to_S_parameter(ABCD3,m, n)
# ABCD4 = S_parameters_to_ABCD(tensor_corner, m, n)
# S4 = ABCD_to_S_parameter(ABCD4,m, n)
# ABCD=ABCD_cascading(ABCD1,ABCD2,ABCD3,m, n)
# S=ABCD_to_S_parameter(ABCD,m, n)
# class reshape(nn.Module):
#     def forward(self,m):
#         m1 = m.view(-1,150)
#         return m1


# 保存到 pickle 文件
# with open("S_parameter.pkl", "wb") as f:
#     pickle.dump(S, f)
# print("S 参数已保存为 S_parameter.pkl 文件")
# 加载 S 参数
with open("S_parameter.pkl", "rb") as f:
    loaded_S = pickle.load(f)

print(loaded_S)
def PEL_test(output2):
    num_samples = output2.shape[0]
    S_cel = torch.cat(
        [torch.complex(output2[:, 0, :], output2[:, 1, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]),torch.complex(output2[:, 0, :], output2[:, 1, :])],
        dim=1)
    S_cel=S_cel.view(num_samples, 4, 1000)
    batch_size, channels, orl = S_cel.shape
    tensor = S_cel.permute(0, 2, 1).reshape(batch_size, orl, 2, 2)
    S = torch.linalg.svdvals(tensor)
    value = S.max(dim=-1).values

    value = torch.where(value > 1, 1/ value, torch.ones_like(value))
    value = value.reshape(batch_size, 1, orl)
    # angle = torch.log(value)
    # angle = torch.cat([angle, torch.flip(angle[:, :, 0:-1], [2])], dim=2)
    # angle = torch.fft.fft(angle, dim=2)
    # angle = torch.cat([angle[:, :, 0].unsqueeze(1), 2.0 * angle[:, :, 1:orl],
    #                    0.0 * 2.0 * angle[:, :, orl:]], dim=2)
    # angle = torch.fft.ifft(angle, dim=2)
    # angle = angle[:, :, :orl]
    # passivity_filter = value * torch.exp(1j * angle)
    S_pel =  S_cel* value
    # S_pel =  S_cel * passivity_filter
    value.reshape(batch_size * orl)
    output2[:, 0, :] = torch.real(S_pel[:, 0, :])
    output2[:, 1, :] = torch.imag(S_pel[:, 0, :])
    output2[:, 2, :] = torch.real(S_pel[:, 1, :])
    output2[:, 3, :] = torch.imag(S_pel[:, 1, :])
    return output2

class PEL(nn.Module):
    def forward(self,output2,batch_size=32):
        num_samples = output2.shape[0]
        output3=torch.zeros((num_samples, 4, 1000))
        S_cel = torch.cat(
            [torch.complex(output2[:, 0, :], output2[:, 1, :]), torch.complex(output2[:, 2, :], output2[:, 3, :]),
             torch.complex(output2[:, 2, :], output2[:, 3, :]), torch.complex(output2[:, 0, :], output2[:, 1, :])],
            dim=1)
        S_cel = S_cel.view(num_samples, 4, 1000)
        batch_size, channels, orl = S_cel.shape
        tensor = S_cel.permute(0, 2, 1).reshape(batch_size, orl, 2, 2)
        S = torch.linalg.svdvals(tensor)
        value = S.max(dim=-1).values
        value = torch.where(value > 1, 1 / value, torch.ones_like(value))
        value = value.reshape(batch_size, 1, orl)

        # angle = torch.log(value)
        # angle = torch.cat([angle, torch.flip(angle[:, :, 0:-1], [2])], dim=2)
        # angle = torch.fft.fft(angle, dim=2)
        # angle = torch.cat([angle[:, :, 0].unsqueeze(1), 2.0 * angle[:, :, 1:orl],
        #                    0.0 * 2.0 * angle[:, :, orl:]], dim=2)
        # angle = torch.fft.ifft(angle, dim=2)
        # angle = angle[:, :, :orl]
        # passivity_filter = value * torch.exp(1j * angle)

        S_pel = S_cel * value
        # S_pel = S_cel * passivity_filter
        value.reshape(batch_size * orl)
        output3[:, 0, :] = torch.real(S_pel[:, 0, :])
        output3[:, 1, :] = torch.imag(S_pel[:, 0, :])
        output3[:, 2, :] = torch.real(S_pel[:, 1, :])
        output3[:, 3, :] = torch.imag(S_pel[:, 1, :])
        return output3
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
        # self.tconv5 = nn.ConvTranspose1d(25, 2, 4, 1, 0)
        self.CEL = CEL()
        self.PEL = PEL()

    def conv(self, x):
        latent = self.conv1(x)
        latent = F.elu(latent)

        latent = self.conv2(latent)
        latent = F.elu(latent)

        latent = self.conv3(latent)
        latent = F.elu(latent)

        latent = self.conv4(latent)
        latent = F.elu(latent)

        latent = self.conv5(latent)
        latent = F.elu(latent)

        latent = self.conv6(latent)
        latent = F.elu(latent)

        latent = self.conv7(latent)
        latent = F.elu(latent)

        return latent

    def transposed_conv(self, z):
        latent = self.tconv1(z)
        latent = F.elu(latent)

        latent = self.tconv2(latent)
        latent = F.elu(latent)

        latent = self.tconv3(latent)
        latent = F.elu(latent)

        latent = self.tconv4(latent)
        latent = F.elu(latent)

        recons_y = self.tconv5(latent)
        return recons_y[..., :1500]

    def forward(self, x, iter=None, total_iter=None):
        z = self.conv(x)
        out1 = self.transposed_conv(z)
        out2 = self.CEL(out1,1500)
        if iter is not None and total_iter is not None and iter >= (total_iter - 500):
            out2 = self.PEL(out2)
        # out2 = torch.clamp(out2, min=-1, max=1)
        return out2

cwd = os.getcwd()

#Device to be used in training. If GPU is available, it will be automatically used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

absolute_path = 'C:/Users/45042/PycharmProjects/pythonProject1/app2.mat'

# 使用绝对路径加载数据
training_data = scipy.io.loadmat(absolute_path)
#Load Training + Test Data. The data is generated using Latin Hypercube Sampling (LHS)

#train_test_y = training_data['va1']
#train_test_y = train_test_y.squeeze()
#train_test_x = training_data['trainingInputs']
#train_test_y=train_test_y[:,[0,1],:]
#train_test_y*=10


# S = S.numpy()
# train_test_y =  training_data['va1']
# train_test_y = train_test_y.squeeze()
train_test_y = loaded_y4
train_test_x = loaded_S.detach().numpy()
# train_test_x=train_test_x[:,[0,1],:]



#Select how many samples to be used for training.
Ndata = train_test_y.shape[0]
Ntrain = 102
Ntest = Ndata-Ntrain
#get train/test indices by linspace to avoid disrupting LHS
# 设置随机种子以确保结果可复现（可选）
np.random.seed(42)

# 随机选择训练集索引
train_indices = np.random.choice(Ndata, size=Ntrain, replace=False)
# train_indices = [73, 118, 4, 52, 65, 60, 21, 89, 48, 23, 162, 161, 196, 193, 182, 150, 35, 137, 5, 77, 45, 197, 194, 123, 51, 9, 17, 127, 129, 135, 136, 72, 144, 99, 132, 199, 37, 192, 70, 33, 50, 14, 183, 130, 149, 174, 2, 160, 195, 62, 44, 69, 147, 168, 13, 64, 41, 28, 142, 170, 190, 148, 83, 36, 15, 42, 79, 177, 169, 84, 31, 88, 158, 57]
# train_indices =[73, 102, 52, 101, 21, 178, 23, 139, 196, 119, 150, 80, 5, 128, 197, 63, 51, 78, 127, 83, 136, 31, 99, 82, 37, 108, 33, 66, 183, 115, 174, 189, 195, 169]
# train_indices =[73, 60, 162, 150, 45, 9, 136, 199, 50, 174, 144, 149, 137, 37, 64, 28, 190, 14, 15, 79, 197, 84, 32, 158]
# train_indices =[73, 60, 162, 150, 45, 9, 136, 199, 50, 174]
# train_indices = np.linspace(0, Ndata-1, num=Ntrain).round().astype(int)
dumm = np.arange(0, Ndata)
test_idx = np.delete(dumm, train_indices)
# indices_to_move=[70,48]
# train_indices = np.append(train_indices, test_idx[indices_to_move])
# test_idx = np.delete(test_idx, indices_to_move)
# train_indices = np.sort(train_indices)
# train_indices.remove(40)
training_x = train_test_x[train_indices, :]
test_x = train_test_x[test_idx, :]

training_y = train_test_y[train_indices, :, :]
test_y = train_test_y[test_idx, :, :]

# # Get normalization values using Training Data.
# meanY = training_y.mean(axis=0)
# stdY = training_y.std(axis=0)
#
# MeanY = torch.tensor(meanY).to(device)
# StdY = torch.tensor(stdY).to(device)
#
# # Inputs are scaled between [-1, 1].
# meanX = train_test_x.min(axis=0)
# stdX = train_test_x.max(axis=0) - train_test_x.min(axis=0)
#
# #Normalize Train and Test Data
# #Use mean and std of train data to normalize test data to avoid bias
# training_x = -1+2*(training_x-meanX)/stdX
# training_y = (training_y-meanY)/stdY
#
# test_x = -1+2*(test_x - meanX)/stdX
# test_y = (test_y - meanY)/stdY
#
# dimIn = training_x.shape[1]
#
# train_test_y_normalized = (train_test_y-meanY)/stdY
# train_test_x_normalized = -1+2*(train_test_x-meanX)/stdX
#
# Convert everything to tensor
tensor_y = torch.Tensor(training_y).to(device)
tensor_x = torch.Tensor(training_x).to(device)

tensor_test_y = torch.Tensor(test_y).to(device)
tensor_test_x = torch.Tensor(test_x).to(device)

print('Done loading data and pre-processing.')


NMSE_Test = []
modelWeights= []
NMSE_Test_Median = []
#Select model. 2 different STCNN types are provided in "models.py". (Un)comment below to choose.
#model = Solenoid_STCNN().to(device)
model = SNN().to(device)

#Select optimizer to be used, define initial "learning rate (lr)", and learning rate reduction ratio (gamma) at milestones.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [500, 1000,1500,2000,2500], gamma=0.5)
numParams = sum([p.numel() for p in model.parameters()])

print(f"Model is loaded. Number of learnable parameters in the network {numParams:d}")


#Training Loss Function. This is different than MSE loss, see the paper for details.
def calc_training_loss(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)/recon_y.shape[2]
    return err.sqrt().mean()

#Calculate Test Accuracy. For each response in the data (L(f), R(f)), calculate NMSE (see paper for details).
def calc_test_NMSE(recon_y, y):
    err = (torch.abs(y-recon_y)**2).sum(dim=2)
    mean_ref = y.mean(dim=2)
    mean_ref = mean_ref.unsqueeze(-1).repeat(1, 1, y.shape[2])
    norm = (torch.abs(y-mean_ref)**2).squeeze().sum(dim=2)
    NMSE = err/norm
    return NMSE

#Closure to be called by optimizer during training.
def closure(data_x, data_y, iter, total_iter):
    optimizer.zero_grad()
    output = model(data_x, iter=iter, total_iter=total_iter)
    loss = calc_training_loss(output, data_y)
    return loss

#Man Training Loop.
#Results are printed at every "test_schedule" epochs.
test_schedule = 5
training_iter = 4000
current_time = time.time()
print(f"Starting training the model. \n")
print(f"""-----------------------------------------------------------------""")

for a in range(training_iter):
    model.train()
    model.zero_grad()
    train_data_x = tensor_x
    train_data_y = tensor_y
    loss = closure(train_data_x, train_data_y, iter=a, total_iter=training_iter)

    loss.backward()
    optimizer.step()
    scheduler.step()
    # Set into eval mode for testing.

    if a % test_schedule == 0:
        model.eval()
        with torch.no_grad():
            test_y = tensor_test_y
            test_x = tensor_test_x
            test_output = model(test_x)
            if a is not None and training_iter is not None and a >= (training_iter - 500):
                test_output = PEL_test(test_output)
            NMSE = calc_test_NMSE(test_output, test_y)
            avNMSE = NMSE.mean()
            medNMSE = NMSE.median()

            print(
                f"Train Iter {(a+1):d}/{training_iter:d} Last Iteration Performance on Test Set - Average NMSE: {avNMSE.item():.5f}, Median NMSE: {medNMSE.item():.5f}")

        # 如果需要，可以保存最后一代的模型权重
        save_name_last_iter = "STCNN_Solenoid1_last_iter.pth"
        torch.save(model.state_dict(), cwd + "/" + save_name_last_iter)
        print(f"Last iteration model weights are saved in \"{cwd}/{save_name_last_iter}\"")



# test_output1= test_output.view(-1,1000).T
# test_y = test_y.view(-1,1000).T
elapsed = time.time() - current_time
print(f"""\n-----------------------------------------------------------------""")
print(f"""Training is completed in {elapsed/60 :.3f} minutes""")

row_sums = NMSE.sum(dim=1)
# Calculate the top 10 highest sum values and their indices
top_values, top_indices = torch.topk(row_sums, 30, largest=True)

# Print the top NMSE sum values and their corresponding indices
print('Top 30 sum NMSE values:', top_values)
print('Indices of the top 30 sum NMSE values:', top_indices)
# 获取所有大于0.08的值的布尔掩码
mask = row_sums > 0.06

# 使用这个布尔掩码来获取所有大于0.08的值
values = row_sums[mask]

# 获取这些值的索引
indices = torch.nonzero(mask).squeeze()

# 打印值和索引
print('Values greater than 0.06:', values)
print('Indices of values greater than 0.06:', indices)

def PEL_test(output2, batch_size=32):
    num_samples = output2.shape[0]
    num_batches = num_samples // batch_size

    Sp = torch.zeros((num_samples, 4, 4, 1000))
    global_max_s = -np.inf  # 初始化一个全局最大奇异值变量

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch = output2[start_idx:end_idx]

        # 将第二维度为 0 的元素存入
        Sp[start_idx:end_idx, 0, 0, :] = batch[:, 0, :]
        Sp[start_idx:end_idx, 1, 1, :] = batch[:, 0, :]
        Sp[start_idx:end_idx, 2, 2, :] = batch[:, 0, :]
        Sp[start_idx:end_idx, 3, 3, :] = batch[:, 0, :]

        # 将第二维度为 1 的元素存入
        Sp[start_idx:end_idx, 0, 2, :] = batch[:, 1, :]
        Sp[start_idx:end_idx, 1, 3, :] = batch[:, 1, :]
        Sp[start_idx:end_idx, 2, 0, :] = -batch[:, 1, :]
        Sp[start_idx:end_idx, 3, 1, :] = -batch[:, 1, :]

        # 将第二维度为 2 的元素存入
        Sp[start_idx:end_idx, 0, 1, :] = batch[:, 2, :]
        Sp[start_idx:end_idx, 1, 0, :] = batch[:, 2, :]
        Sp[start_idx:end_idx, 2, 3, :] = batch[:, 2, :]
        Sp[start_idx:end_idx, 3, 2, :] = batch[:, 2, :]

        # 将第二维度为 3 的元素存入
        Sp[start_idx:end_idx, 0, 3, :] = batch[:, 3, :]
        Sp[start_idx:end_idx, 1, 2, :] = batch[:, 3, :]
        Sp[start_idx:end_idx, 2, 1, :] = -batch[:, 3, :]
        Sp[start_idx:end_idx, 3, 0, :] = -batch[:, 3, :]
        sigma_max1 = np.zeros((num_samples, 1000))

        for i in range(start_idx, end_idx):
            for j in range(1000):
                # 提取当前 4x4 矩阵
                sub_matrix = Sp[i, :, :, j]
                sub_matrix = sub_matrix.numpy()
                U, s, Vh = np.linalg.svd(sub_matrix)
                # 计算奇异值的最大值，并与全局最大值进行比较
                max_s = np.max(s)
                if max_s > global_max_s:
                    global_max_s = max_s  # 如果更大，则更新全局最大奇异值

                # 输出或返回整体的最大奇异值
            print(global_max_s)
            return global_max_s

sigma_max_test=PEL_test(test_output)

# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, test_output[4,2,:], label='TCNN (Column 1)',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, test_y[4,2,:], label='HFSS (Column 1)',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, test_output[4,3,:], label='TCNN ',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, test_y[4,3,:], label='HFSS',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, test_output[4,0,:], label='TCNN (Column 1)',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, test_y[4,0,:], label='HFSS (Column 1)',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, test_output[4,1,:], label='TCNN ',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, test_y[4,1,:], label='HFSS',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()

# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, S[0,2,:], label='TCNN ', marker='o',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, tensor_test_y4[0,2,:], label='HFSS', marker='*',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, S[0,3,:], label='TCNN', marker='o',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, tensor_test_y4[0,3,:], label='HFSS', marker='*',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, S[0,0,:], label='TCNN ', marker='o',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x,tensor_test_y4[0,0,:], label='HFSS', marker='*',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()
#
# x=torch.arange(1,1001)
# plt.figure(1, figsize=(10, 10))
# plt.plot(x, S[0,1,:], label='TCNN', marker='o',color='blue')
# # 绘制原始数据的目标值的第一列
# plt.plot(x, tensor_test_y4[0,1,:], label='HFSS', marker='*',color='red')
#
# plt.xlabel('frequency')
# plt.ylabel('dB')
# plt.legend()
# plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[1,2,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[1,2,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[1,3,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[1,3,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[1,0,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[1,0,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[1,1,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[1,1,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[2,2,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[2,2,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[2,3,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[2,3,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[2,0,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[2,0,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[2,1,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[2,1,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[70,2,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[70,2,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[70,3,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[70,3,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[70,0,:], label='TCNN ', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[70,0,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()

x=torch.arange(1,1001)
plt.figure(1, figsize=(10, 10))
plt.plot(x, test_output[70,1,:], label='TCNN', marker='o',color='blue')
# 绘制原始数据的目标值的第一列
plt.plot(x, test_y[70,1,:], label='HFSS', marker='*',color='red')

plt.xlabel('frequency')
plt.ylabel('dB')
plt.legend()
plt.show()