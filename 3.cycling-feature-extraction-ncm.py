import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

if __name__== "__main__":
    files= os.listdir("battery-data-ncm-cells")
    feature_name_ls= ["mean", "std", "var", "skew", "kurt", "rms", "max", "min", "median", "p2p"]
    df_res= pd.DataFrame(columns= ['Cell', 'Cycle', 'Voltages', 'Rate', 'Tem', 'Capacity'])
    TPidx, PSPidx, InitialCapacity= 0.90, 0.86, 3500
    specific_fiiles= []; Tem= 45; C_rate= 5; D_rate= 1; count= 0
    rate= f"{C_rate}" if C_rate== 1 else f"0{C_rate}"
    charging_policy= f"CY{Tem}-{rate}_{D_rate}"
    feature_elaxation= pd.DataFrame()
    for file in files:
        fail_count= 0
        if charging_policy in file:
            plt.figure(figsize= (6, 6))
            # plt.title(file, fontsize= 20)
            max_relexation_voltage, min_relexation_voltage= -np.inf, np.inf
            data_r= pd.read_csv(os.path.join("battery-data-ncm-cells", file))
            for cycle in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+1):
                data_i= data_r[data_r['cycle number'] == cycle]
                Ecell = np.array(data_i['Ecell/V'])
                Q_dis = np.array(data_i['Q discharge/mA.h'])
                Current = np.array(data_i['<I>/mA'])
                Control = np.array(data_i['control/V/mA'])
                Cr = np.array(data_i['control/mA'])[1]/ 3500
                if np.max(Q_dis) < 2500:
                    continue
                index = np.where(np.abs(Control) == 0)
                if index[0][0] > 0:
                    start= index[0][0]
                else:
                    start = index[0][14]
                if Control[start + 13] == 0 and np.max(Q_dis)< InitialCapacity* TPidx and np.max(Q_dis)> InitialCapacity* PSPidx:
                    count+= 1
                    relaxation_voltage= pd.DataFrame({"Voltage": Ecell[start: start+ 14]})
                    if max_relexation_voltage< np.max(relaxation_voltage): max_relexation_voltage= np.max(relaxation_voltage)
                    if min_relexation_voltage> np.min(relaxation_voltage): min_relexation_voltage= np.min(relaxation_voltage)
                    MEAN, STD, VAR, SKEW, KURT, RMS= relaxation_voltage.mean(), relaxation_voltage.std(), relaxation_voltage.var(), relaxation_voltage.skew(), relaxation_voltage.kurt(), (relaxation_voltage**2).mean()** (1/2)
                    MAX, MIN, MEDIAN, PEAK2PEAK= relaxation_voltage.max(), relaxation_voltage.min(), relaxation_voltage.median(), relaxation_voltage.max()- relaxation_voltage.min()
                    feature_elaxation_voltage= pd.concat([MEAN, STD, VAR, SKEW, KURT, RMS, MAX, MIN, MEDIAN, PEAK2PEAK])
                    feature_elaxation_voltage.index= np.array([["V-"+ f_name] for f_name in feature_name_ls]).ravel()
                    feature_elaxation_voltage= feature_elaxation_voltage.to_frame().T
                    feature_elaxation_voltage["Charging-Policy"]= [charging_policy]
                    feature_elaxation_voltage["Cell-Name"]= [charging_policy+ "-"+ file.split(".")[0].split("-")[-1]]
                    feature_elaxation_voltage["Cycle"]= [cycle]
                    feature_elaxation= pd.concat([feature_elaxation, feature_elaxation_voltage], axis= 0, ignore_index= 0)
                    plt.plot(np.arange(0, len(Ecell[start: start+ 14]))* 120, Ecell[start: start+ 14])
                    df_res= pd.concat([df_res, pd.DataFrame({'Cell': file.split("#")[-1].split(".")[0], 
                                                                                            'Cycle': cycle, 
                                                                                            'Voltages': Ecell[start: start+ 14], 
                                                                                            'Rate': Cr, 
                                                                                            'Tem': Tem, 
                                                                                            'Capacity': np.max(Q_dis)})], 
                                                                                            axis= 0, 
                                                                                            ignore_index= True)
            if min_relexation_voltage!= np.inf or max_relexation_voltage!= -np.inf:
                plt.ylim(min_relexation_voltage, max_relexation_voltage)
            plt.xlim(0, len(Ecell[start: start+ 14])* 120)
            plt.xticks(fontsize= 15)
            plt.yticks(fontsize= 15)
            plt.xlabel("Time (s)", fontsize= 15)
            plt.ylabel("Voltage (V)", fontsize= 15)
            plt.grid(True)
            plt.show()
        feature_elaxation= feature_elaxation.reset_index(drop= True)
        # feature_elaxation.to_csv("feature-elaxation-ncm-tem%s-Crate%s-Drate%s.csv"%(Tem, C_rate, D_rate), index= False)