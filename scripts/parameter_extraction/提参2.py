import matplotlib.pyplot as plt
import numpy as np
import skrf
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from scipy.signal import medfilt
import re
import os
from pathlib import Path
import pandas as pd
from skrf import Frequency, Network


def path_S2P(path):
    mom_data = skrf.Network(path)
    S_all=mom_data.s[:300,:,:]
    S11, S12, S21, S22 = S_all[:, 0, 0], S_all[:, 0, 1], S_all[:, 1, 0], S_all[:, 1, 1]
    Y_all = skrf.s2y(S_all)
    Y11, Y12, Y21, Y22 = Y_all[:, 0, 0], Y_all[:, 0, 1], Y_all[:, 1, 0], Y_all[:, 1, 1]
    freq = mom_data.f[:300]
    return S11, S12,S21, S22, Y11, Y12, Y21, Y22, freq

def S_ABCD(S11,S12,S21,S22):
    A = ((1 + S11) * (1 - S22) + S12 * S21) / (2 * S21)
    B = 50 * ((1 + S11) * (1 + S22) - S12 * S21) / (2 * S21)
    C = 1 / 50 * ((1 - S11) * (1 - S22) - S12 * S21) / (2 * S21)
    D = ((1 - S11) * (1 + S22) + S12 * S21) / (2 * S21)
    return  A, B, C, D

def ABCD_RLGC(A,B,C,D,freq,l):
    Z0 = np.sqrt(B / C)
    Gamma=np.arccosh((A+D)/2)
    # Gamma=np.arccosh(A)#
    # Gamma=np.arcsinh(np.sqrt(B*C))
    #########拼Gamma虚部##################
    RGamma,IGamma=np.real(Gamma),np.imag(Gamma)
    index=1
    while(index):
        index=0
        for k in range(len(IGamma)-1):
            if(IGamma[k+1]<IGamma[k]-np.pi):
                IGamma[k+1]=IGamma[k+1]+2*np.pi
                index=1
    for k in range(len(IGamma)):
        Gamma[k]=complex(RGamma[k],IGamma[k])
    ###################################
    R=np.real(Z0*Gamma)/l
    G=np.real(Gamma/Z0)/l
    L=np.imag(Z0*Gamma)/l/(2*np.pi*freq)
    C_=np.imag(Gamma/Z0)/l/(2*np.pi*freq)
    # for k in range(len(A)):
        # L[k]=L[k]/(2*np.pi*(0.2*k+0.2)*1e9)
        # C_[k]=C_[k]/(2*np.pi*(0.2*k+0.2)*1e9)
    return R,L,G,C_,Z0,Gamma

def RLGC_SPICE_rlgc_way3(R,L,G,C,l,freq,p1=None,p2=None):
    if p1 is None:
        p1 = 0
    if p2 is None:
        p2 = len(freq)-1
    eps = 1e-30
    r3_ratio_fixed = 0.70

    def unpack_RL_params(k1):
        Rlow, dR = k1[0], k1[1]
        L1 = k1[2] * 1e-9
        tau2 = k1[3] * 1e-12
        tau3 = k1[4] * 1e-12

        R1 = Rlow + dR
        R3 = R1 * r3_ratio_fixed / (1 - r3_ratio_fixed)
        R2 = Rlow * R1 / dR
        L2 = tau2 * (R1 + R2)
        L3 = tau3 * R3
        return R1, R2, R3, L1, L2, L3, Rlow, dR, tau2, tau3

    def error_RL(k1,R,L):
        r,l=[],[]
        R1,R2,R3,L1,L2,L3,Rlow,dR,tau2,tau3=unpack_RL_params(k1)
        for k in range(len(freq)):#len(R)
            Omega=2*np.pi*freq[k]
            x2=(Omega*tau2)**2
            x3=(Omega*tau3)**2
            R_RLGC=(Rlow+R1*x2)/(1+x2)+R3*x3/(1+x3)
            L_RLGC=tau2*dR/(1+x2)+tau3*R3/(1+x3)+L1
            r.append(R_RLGC)
            l.append(L_RLGC)
        error=[]
        R_scale=max(abs(R[p2]), eps)
        L_scale=max(abs(L[p2]), eps)
        for kk in range(p1,p2):#
            # if(not(40<=kk<=50)):
            error1=(r[kk]-R[kk])/R_scale
            error.append(error1)
        for kk in range(p1,p2):
            # if(not(40<=kk<=50)):
            error2=(l[kk]-L[kk])/L_scale
            error.append(error2)
        error.append(0.05*max(0, np.log10(tau3/max(tau2, eps))))
        error.append(1.0*max(0, np.log10(R2/15000)))
        return error
    Rdc,Ldc,Gdc,Cdc=R[0],L[0],G[0],C[0]
    Rhf,Lhf,Ghf,Chf=R[p2],L[p2],G[p2],C[p2]

    # 先优化更稳定且可辨识的量，再反算回原电路元件：
    # Rlow = R1 || R2，dR = R1 - Rlow，tau2 = L2/(R1+R2)，tau3 = L3/R3。
    # 固定 R3/(R1+R3)，避免 R1 和 R3 对同一个高频电阻贡献产生多组等价分配。
    Rlow=abs(Rdc)
    R1=max((1-r3_ratio_fixed)*abs(Rhf), Rlow*1.2)
    R3=R1*r3_ratio_fixed/(1-r3_ratio_fixed)
    dR=max(R1-Rlow, Rlow*0.05, 1e-9)
    L1=abs(Lhf)
    L3_init=abs(Ldc-L[p2//2])
    L2_low=abs(Ldc-Lhf-L3_init)
    tau2=max(L2_low/dR, 1e-13)
    tau3=max(L3_init/max(R3, eps), 1e-13)

    R2_soft_max = 1.5e4
    dR_min=max(Rlow**2/max(R2_soft_max-Rlow, Rlow), Rlow*1e-3, 1e-9)
    dR=max(dR, dR_min)
    bounds1 = ([max(Rlow*0.2, eps), dR_min, max(L1*0.25*1e9, eps), 0.1, 0.1],
       [max(Rlow*5, eps), max(abs(Rhf)*10, Rlow*10, 100), max(L1*5*1e9, eps), 1000, 500])
    k1 = np.array([Rlow,dR,L1*1e9,tau2*1e12,tau3*1e12])
    k1 = np.minimum(np.maximum(k1, bounds1[0]), bounds1[1])
    Para = least_squares(error_RL, k1,bounds=bounds1,args=(R,L))
    residuals = Para.fun
    rmse = np.sqrt(np.mean(residuals**2))
    R1,R2,R3,L1,L2,L3,_,_,_,_=unpack_RL_params(Para.x)

    def error_GC(k1,G,C):
        gg,cc=[],[]
        Cox,Csi,Rsi=k1[0]*1e-12,k1[1]*1e-12,k1[2]
        for k in range(len(freq)):#len(R)
            
            Omega=2*np.pi*freq[k]
            G_RLGC=(Omega**2*Rsi*(Cox**2))/(1+(Omega**2)*(Rsi**2)*(Cox+Csi)**2)
            C_RLGC=(Cox+(Omega**2)*Csi*(Rsi**2)*Cox*(Cox+Csi))/(1+(Omega**2)*(Rsi**2)*(Cox+Csi)**2)
            gg.append(G_RLGC)
            cc.append(C_RLGC)
        error=[]
        for kk in range(p1,p2):#
            # if(not(40<=kk<=50)):
            error1=(gg[kk]-G[kk])/G[-1]
            error.append(error1)
        for kk in range(p1,p2):
            # if(not(40<=kk<=50)):
            error2=(cc[kk]-C[kk])/C[p2]
            error.append(error2)
        return error

    C1=abs(Cdc)
    C2=abs(Cdc*Chf/(Cdc-Chf))
    Rsi=abs(C1**2/(Ghf*(C1+C2)**2))
    
    # print("R1", R1, bounds1[0][0], bounds1[1][0])
    # print("R2", R2, bounds1[0][1], bounds1[1][1])
    # print("R3", R3, bounds1[0][2], bounds1[1][2])
    # print("L1", L1, bounds1[0][3], bounds1[1][3])
    # print("L2", L2, bounds1[0][4], bounds1[1][4])
    # print("L3", L3, bounds1[0][5], bounds1[1][5])


    bounds1 = ([C1*0.2*1e12, C2*0.2*1e12, Rsi*0.2],
       [C1*5*1e12, C2*5*1e12, Rsi*5])    
    k1 = np.array([C1*1e12, C2*1e12, Rsi])
    # print(bounds1)
    Para = least_squares(error_GC, k1,bounds=bounds1,args=(G,C))
    residuals = Para.fun
    rmse = np.sqrt(np.mean(residuals**2))
    Cox,Csi,Rsi=Para.x
    Cox=Cox*1e-12
    Csi=Csi*1e-12

    # print("Cox", Cox, bounds1[0][0], bounds1[1][0])
    # print("Csi", Csi, bounds1[0][1], bounds1[1][1])
    # print("Rsi", Rsi, bounds1[0][2], bounds1[1][2])
    # print("#######################")

    parameter_spice=np.empty(9)
    values = [R1,R2,R3,L1*1e9,L2*1e9,L3*1e9,Cox*1e12,Csi*1e12,Rsi]
    # print(values)
    for kk in range(len(values)):
        parameter_spice[kk]=values[kk]


    #计算得到S参数
    R_all,L_all,G_all,C_all=[],[],[],[]
    S11_all,S12_all,S21_all,S22_all=[],[],[],[]
    Sps=[]
    Yps=[]
    for f in freq:#
        j=1j
        Omega=2*np.pi*f
        R_RLGC=(R1**2 * R2 + R1 * R2**2 + Omega**2 * R1 * L2 ** 2) / ((R1 + R2) ** 2 + Omega ** 2 * L2 ** 2)+(Omega**2*L3**2*R3)/(R3**2+Omega**2*L3**2)
        L_RLGC=(R1**2*L2)/((R1+R2)**2+Omega**2*L2**2)+L3*R3**2/(R3**2+Omega**2*L3**2)+L1
        G_RLGC=(Omega**2*Rsi*(Cox**2))/(1+(Omega**2)*(Rsi**2)*(Cox+Csi)**2)
        C_RLGC=(Cox+(Omega**2)*Csi*(Rsi**2)*Cox*(Cox+Csi))/(1+(Omega**2)*(Rsi**2)*(Cox+Csi)**2)
        Z0 = np.sqrt((R_RLGC + j * Omega * L_RLGC) / (G_RLGC + j * Omega * C_RLGC))
        GAMMA=np.sqrt((R_RLGC+j*Omega*L_RLGC)*(G_RLGC+j*Omega*C_RLGC))
        A=np.cosh(GAMMA*l)
        B=Z0*np.sinh(GAMMA*l)
        C=1/Z0*np.sinh(GAMMA*l)
        D=np.cosh(GAMMA*l)
        S11=(A+B/50-C*50-D)/(A+B/50+C*50+D)
        S12=2*(A*D-B*C)/(A+B/50+C*50+D)
        S21=2/(A+B/50+C*50+D)
        S22=(-A+B/50-C*50+D)/(A+B/50+C*50+D)
        R_all.append(R_RLGC)
        L_all.append(L_RLGC)
        G_all.append(G_RLGC)
        C_all.append(C_RLGC)
        S11_all.append(S11)
        S12_all.append(S12)
        S21_all.append(S21)
        S22_all.append(S22)
        Sps.append([[S11,S12],[S21,S22]])
    s_ntw=Network(frequency=freq, s=Sps)
    # Yps=skrf.s2y(s_ntw.s)
    # Y11_all, Y12_all, Y21_all, Y22_all = Yps[:, 0, 0], Yps[:, 0, 1], Yps[:, 1, 0], Yps[:, 1, 1]
    # return S11_all,S12_all,S21_all,S22_all,Y11_all, Y12_all, Y21_all, Y22_all,R_all,L_all,C_all,G_all,parameter_spice,rmse
    return S11_all,S12_all,S21_all,S22_all,R_all,L_all,C_all,G_all,parameter_spice,rmse


def plot_2ports_Leq(Yparameters,names,freqs):
    fig, axes = plt.subplots(1, 2)  # 4 行 2 
    for idx, (Yp, freq, name) in enumerate(zip(Yparameters, freqs, names)):
        # Leq=[]
        # for f in (freq):
            # Leq.append(np.imag(1/Yp[0,0])/2/np.pi/f)
        Leq=(np.imag(1/Yp[0,0])/2/np.pi/freq)
        Q = np.imag(1/Yp[0,0]) / np.real(1/Yp[0,0])
        axes[0].plot(freq, Leq, label=f"{name}")
        axes[1].plot(freq, Q, label=f"{name}")

    axes[0].set_title("Equivalent Inductance", fontsize=12)
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].legend()

    axes[1].set_title("Quality Factor", fontsize=12)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].legend()
    # plt.savefig("L_plot.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    # plt.show()


def plot_2ports_Req(Yparameters,names,freqs):
    fig, axes = plt.subplots(1, 2)  # 4 行 2 
    for idx, (Yp, freq, name) in enumerate(zip(Yparameters, freqs, names)):
        # Leq=[]
        # for f in (freq):
            # Leq.append(np.imag(1/Yp[0,0])/2/np.pi/f)
        Req=(np.real(1/Yp[0,0]))
        Q = np.imag(1/Yp[0,0]) / np.real(1/Yp[0,0])
        axes[0].plot(freq, Req, label=f"{name}")
        axes[1].plot(freq, Q, label=f"{name}")

    axes[0].set_title("Equivalent Resistance", fontsize=12)
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].legend()

    axes[1].set_title("Quality Factor", fontsize=12)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].legend()
    # plt.savefig("L_plot.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    # plt.show()



def plot_2ports_S(Sparameters,names,type,freqs):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 4 行 2 列

    if type == "RI":
        titles = [
            "S11 Real", "S11 Imag", "S12 Real", "S12 Imag",
            "S21 Real", "S21 Imag", "S22 Real", "S22 Imag"
        ]
        ylabels = ["Real"] * 4 + ["Real"] * 4
    elif type == "MP":
        titles = [
            "S11 |dB|", "S11 Phase", "S12 |dB|", "S12 Phase",
            "S21 |dB|", "S21 Phase", "S22 |dB|", "S22 Phase"
        ]
        ylabels = ["dB", "deg"] * 4
    else:
        raise ValueError(f"Unknown type: {type}")

    if type == "RI":
        for idx, (Sp, name, freq) in enumerate(zip(Sparameters, names, freqs)):
            axes[0, 0].plot(freq, Sp[0,0].real, label=f"{name}")
            axes[0, 0].legend()
            axes[0, 1].plot(freq, Sp[0,0].imag, label=f"{name}")
            axes[0, 1].legend()
            axes[0, 2].plot(freq, Sp[0,1].real, label=f"{name}")
            axes[0, 2].legend()
            axes[0, 3].plot(freq, Sp[0,1].imag, label=f"{name}")
            axes[0, 3].legend()
            axes[1, 0].plot(freq, Sp[1,0].real, label=f"{name}")
            axes[1, 0].legend()
            axes[1, 1].plot(freq, Sp[1,0].imag, label=f"{name}")
            axes[1, 1].legend()
            axes[1, 2].plot(freq, Sp[1,1].real, label=f"{name}")
            axes[1, 2].legend()
            axes[1, 3].plot(freq, Sp[1,1].imag, label=f"{name}")
            axes[1, 3].legend()

    if type == "MP":
        for idx, (Sp, name, freq) in enumerate(zip(Sparameters, names, freqs)):
            axes[0, 0].plot(freq, 20*np.log10(abs(Sp[0,0])), label=f"{name}")
            axes[0, 0].legend()
            axes[0, 1].plot(freq, np.angle(Sp[0,0],deg=True), label=f"{name}")
            axes[0, 1].legend()
            axes[0, 2].plot(freq, 20*np.log10(abs(Sp[0,1])), label=f"{name}")
            axes[0, 2].legend()
            axes[0, 3].plot(freq, np.angle(Sp[0,1],deg=True), label=f"{name}")
            axes[0, 3].legend()
            axes[1, 0].plot(freq, 20*np.log10(abs(Sp[1,0])), label=f"{name}")
            axes[1, 0].legend()
            axes[1, 1].plot(freq, np.angle(Sp[1,0],deg=True), label=f"{name}")
            axes[1, 1].legend()
            axes[1, 2].plot(freq, 20*np.log10(abs(Sp[1,1])), label=f"{name}")
            axes[1, 2].legend()
            axes[1, 3].plot(freq, np.angle(Sp[1,1],deg=True), label=f"{name}")
            axes[1, 3].legend()

    # 添加标题、坐标轴标签、图例
    for i in range(2):
        for j in range(4):
            axes[i, j].set_title(titles[i * 4 + j], fontsize=12)
            axes[i, j].set_xlabel("Frequency (GHz)")
            axes[i, j].set_ylabel(ylabels[i * 4 + j])
            axes[i, j].legend()

    fig.suptitle(f"S-Parameter Plot ({type})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间
    # plt.savefig("S_plot.png", dpi=300, bbox_inches='tight')
    # plt.show()
    


def plot_RLGC(RLGCs,freqs,names):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 行 2 列

    titles = ["R", "L", "G", "C"]
    ylabels = ["Resistance (Ohm)", "Inductance (H)", "Conductance (S)", "Capacitance (F)"]

    for idx, (RLGC, freq,name) in enumerate(zip(RLGCs, freqs,names)):
        axes[0,0].plot(freq, RLGC[0], label=f"{name}")
        axes[0,0].set_title(titles[0])
        axes[0,0].set_xlabel("Frequency (GHz)")
        axes[0,0].legend()
        axes[0,1].plot(freq, RLGC[1], label=f"{name}")
        axes[0,1].set_title(titles[1])
        axes[0,1].set_xlabel("Frequency (GHz)")
        axes[0,1].legend()
        axes[1,0].plot(freq, RLGC[2], label=f"{name}")
        axes[1,0].set_title(titles[2])
        axes[1,0].set_xlabel("Frequency (GHz)")
        axes[1,0].legend()
        axes[1,1].plot(freq, RLGC[3], label=f"{name}")
        axes[1,1].set_title(titles[3])
        axes[1,1].set_xlabel("Frequency (GHz)")
        axes[1,1].legend()

    plt.tight_layout()
    # plt.show()
    

def find_resonation_frequency(Y11):
    Q = np.imag(1/Y11) / np.real(1/Y11)
    for idx,x in enumerate(Q):
        if x < 0:
            break
    return idx



#######主函数部分##########################
if __name__ == '__main__':

    os.chdir(Path(__file__).resolve().parents[2])  # 切换到脚本所在目录（可选）
    # output_csv_path = r"D:\MLIN\SNP\output_variables_real_0628.csv"
        #输入1-150Ghz长度为L的EM数据，长度为100um的EM数据,L的长度
    # 初始化 CSV 数据存储列表
    csv_data = []

    # 处理多个文件 dut0.s4p 到 dut3000.s4p
    for i in range(0, 300, 1):  
        momentum_path = fr"./data/sparameters/RDL_Bottom_Snp/dut{i}.s2p"  # 请确保路径与您实际存放的路径一致
        
        if not os.path.exists(momentum_path):  
            print(f"文件 {momentum_path} 不存在，跳过")
            continue
    
        variables = {}
    
        # 1. 动态读取并解析 s2p 头部注释中的变量
        with open(momentum_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # 遇到 "#" 说明注释块结束，进入了 S 参数数据区，直接跳出循环节省算力
                if line.startswith('#'):
                    break
                    
                if line.startswith('!'):  
                    line = line[1:].strip()  # 去掉 "!"
                    if '=' in line:
                        var_name, rest = line.split('=', 1)
                        var_name = var_name.strip()
                        rest = rest.strip()
                        
                        # 正则匹配提取浮点数
                        match = re.search(r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', rest)
                        if match:
                            value = float(match.group(1))
                            variables[var_name] = value
        
        # 2. 根据 RDL_top 的变量名获取具体数值
        # 提取出来的数值默认不带单位（比如 "100um" 提取出来是浮点数 100.0）

        l_rdl = variables.get('ldown')
        w_rdl = variables.get('wdown')
        t_rdl = variables.get('tdown')
        h_tsv = variables.get('htsv')
        p_rdl = variables.get('p1')
        

        parameter_L = l_rdl*1e-6


        S11, S12,S21, S22, Y11, Y12, Y21, Y22, freq = path_S2P(momentum_path)
        yp1=np.array([[Y11, Y12],[Y21, Y22]])
        sp1=np.array([[S11, S12],[S21, S22]])
        
        p2=len(freq)-1
        p1=0

        A, B, C, D=S_ABCD(S11,S12,S21,S22)
        R_l,L_l,G_l,C_l,Z0,Gamma=ABCD_RLGC(A,B,C,D,freq,parameter_L)
        A, B, C, D=S_ABCD(-S11,S12,S21,-S22)
        # _,_,G_l,C_l,Z0,Gamma=ABCD_RLGC(A,B,C,D,freq,parameter_L)
        


        RLGC1=[R_l,L_l,G_l,C_l]

        S11,S12,S21,S22,R_all,L_all,C_all,G_all,parameter_spice,rmse=RLGC_SPICE_rlgc_way3(R_l,L_l,G_l,C_l,parameter_L,freq,p1=p1,p2=p2)
        sp2=np.array([[S11, S12],[S21, S22]]) 
        RLGC2=[R_all,L_all,G_all,C_all]

        sps = [sp1, sp2]
        freqs = [freq, freq]       
        RLGCs=[RLGC1, RLGC2]
 


        # # 绘图
        # plot_RLGC(RLGCs=RLGCs, freqs=freqs, names=["Simulated","Fitting"])
        # plot_2ports_S(Sparameters=sps, names=["Simulated","Fitting"],type="RI",freqs=freqs)        
        # plot_2ports_S(Sparameters=sps, names=["Simulated","Fitting"],type="MP",freqs=freqs)        
        # # plot_2ports_Req(Yparameters=yps, names=["Simulated","Fitting"],freqs=freqs)
        # plt.show()
        

        # 导出提取得到的参数
        output_csv_path = r"./data/tables/RDL_Bottom_TD_3.csv" 
        csv_row = [l_rdl, w_rdl, t_rdl, h_tsv, p_rdl]+list(parameter_spice)+[rmse]
        csv_data.append(csv_row)
        csv_headers = ["l_rdl", "w_rdl", "t_rdl", "h_tsv", "p_rdl", "R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi",  "Rsi","rmse"]
        # csv_headers = ["d_tsv", "h_tsv", "p_rdl", "R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi",  "Rsi","rmse"]
        df = pd.DataFrame(csv_data, columns=csv_headers)
        df.to_csv(output_csv_path, index=False)
        print(f"dut{i}参数保存到：{output_csv_path}")

