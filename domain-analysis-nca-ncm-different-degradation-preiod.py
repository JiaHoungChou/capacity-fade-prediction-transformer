import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.rcParams["font.family"] = "Times New Roman"

def ReadDatasetsAndMergeDatasets(folder_name: str):
    filename_ls= os.listdir(os.path.join(os.getcwd(), folder_name))
    feature_data= pd.DataFrame()
    for filename in filename_ls:
        data= pd.read_csv(os.path.join(os.getcwd(), folder_name, filename))
        data["Charging-Policy"]= filename.split("-")[2].upper()+ " "+ data["Charging-Policy"]
        feature_data= pd.concat([feature_data, data], axis= 0)
    feature_data= feature_data.reset_index(drop= True)
    return feature_data

def FeatureExtraction_NCA_Dataset(dataset_path: str, Tem: int, C_rate: int, D_rate: int, TPidx: float, PSPidx: float, InitialCapacity: int):
    files= os.listdir(dataset_path)
    feature_name_ls= ["mean", "std", "var", "skew", "kurt", "rms", "max", "min", "median", "p2p"]
    df_res= pd.DataFrame(columns= ['Cell', 'Cycle', 'Voltages', 'Rate', 'Tem', 'Capacity'])
    count= 0
    rate= f"{C_rate}" if C_rate== 1 else f"0{C_rate}"
    charging_policy= f"CY{Tem}-{rate}_{D_rate}"
    feature_elaxation= pd.DataFrame()
    for file in files:
        if charging_policy in file:
            max_relexation_voltage, min_relexation_voltage= -np.inf, np.inf
            data_r= pd.read_csv(os.path.join(dataset_path, file))
            for cycle in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+ 1):
                data_i= data_r[data_r['cycle number']== cycle]
                Ecell= np.array(data_i['Ecell/V'])
                Q_dis= np.array(data_i['Q discharge/mA.h'])
                Current= np.array(data_i['<I>/mA'])
                Control= np.array(data_i['control/V/mA'])
                Cr= np.array(data_i['control/mA'])[1]/ 3500
                if np.max(Q_dis) < 2500 or np.max(Q_dis) > 3500:
                    continue
                start, end= np.where(np.abs(Control) == 0)[0][0], 13
                for j in range(0, 3):
                    if Control[start+ 3] == 0:
                        break
                    else:
                        start= np.where(np.abs(Control) == 0)[0][j+1]
                if Current[start] > 1:
                    start= start + 1
                    if Control[start + 13] != 0:
                        end= 12
                if Control[start+ end] == 0 and Ecell[start+ end] > 4.0 and np.max(Q_dis)< InitialCapacity* TPidx and np.max(Q_dis)> InitialCapacity* PSPidx:
                    count+= 1
                    relaxation_voltage= pd.DataFrame({"Voltage": Ecell[start: start+ 14]})
                    if max_relexation_voltage< np.max(relaxation_voltage): 
                        max_relexation_voltage= np.max(relaxation_voltage)
                    if min_relexation_voltage> np.min(relaxation_voltage): 
                        min_relexation_voltage= np.min(relaxation_voltage)
                    MEAN, STD, VAR, SKEW, KURT, RMS= relaxation_voltage.mean(), relaxation_voltage.std(), relaxation_voltage.var(), relaxation_voltage.skew(), relaxation_voltage.kurt(), (relaxation_voltage**2).mean()** (1/2)
                    MAX, MIN, MEDIAN, PEAK2PEAK= relaxation_voltage.max(), relaxation_voltage.min(), relaxation_voltage.median(), relaxation_voltage.max()- relaxation_voltage.min()
                    feature_elaxation_voltage= pd.concat([MEAN, STD, VAR, SKEW, KURT, RMS, MAX, MIN, MEDIAN, PEAK2PEAK])
                    feature_elaxation_voltage.index= np.array([["V-"+ f_name] for f_name in feature_name_ls]).ravel()
                    feature_elaxation_voltage= feature_elaxation_voltage.to_frame().T
                    feature_elaxation_voltage["Charging-Policy"]= [charging_policy]
                    feature_elaxation_voltage["Cell-Name"]= [charging_policy+ "-"+ file.split(".")[0].split("-")[-1]]
                    feature_elaxation_voltage["Cycle"]= [cycle]
                    feature_elaxation= pd.concat([feature_elaxation, feature_elaxation_voltage], axis= 0, ignore_index= 0)
                    df_res= pd.concat([df_res, pd.DataFrame({'Cell': file.split("#")[-1].split(".")[0], 
                                                                                          'Cycle': cycle, 
                                                                                          'Voltages': Ecell[start: start+ 14], 
                                                                                          'Rate': Cr, 
                                                                                          'Tem': Tem, 
                                                                                          'Capacity': np.max(Q_dis)})], 
                                                                                          axis= 0, 
                                                                                          ignore_index= True)
            feature_elaxation= feature_elaxation.reset_index(drop= True)
            feature_elaxation.to_csv(os.path.join(os.getcwd(), "data", "feature-elaxation-nca-tem%s-Crate%s-Drate%s.csv"%(Tem, C_rate, D_rate)), index= False)
    # print("NCA in Tem: %s, C-rate: %s, D-rate: %s is finished !!"%(Tem, C_rate, D_rate))

def FeatureExtraction_NCM_Dataset(dataset_path: str, Tem: int, C_rate: int, D_rate: int, TPidx: float, PSPidx: float, InitialCapacity: int):
    files= os.listdir(dataset_path)
    feature_name_ls= ["mean", "std", "var", "skew", "kurt", "rms", "max", "min", "median", "p2p"]
    df_res= pd.DataFrame(columns= ['Cell', 'Cycle', 'Voltages', 'Rate', 'Tem', 'Capacity'])
    count= 0
    rate= f"{C_rate}" if C_rate== 1 else f"0{C_rate}"
    charging_policy= f"CY{Tem}-{rate}_{D_rate}"
    feature_elaxation= pd.DataFrame()
    for file in files:
        if charging_policy in file:
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
                    df_res= pd.concat([df_res, pd.DataFrame({'Cell': file.split("#")[-1].split(".")[0], 
                                                                                            'Cycle': cycle, 
                                                                                            'Voltages': Ecell[start: start+ 14], 
                                                                                            'Rate': Cr, 
                                                                                            'Tem': Tem, 
                                                                                            'Capacity': np.max(Q_dis)})], 
                                                                                            axis= 0, 
                                                                                            ignore_index= True)
            feature_elaxation= feature_elaxation.reset_index(drop= True)
            feature_elaxation.to_csv(os.path.join(os.getcwd(), "data", "feature-elaxation-ncm-tem%s-Crate%s-Drate%s.csv"%(Tem, C_rate, D_rate)), index= False)
    # print("NCM in Tem: %s, C-rate: %s, D-rate: %s is finished !!"%(Tem, C_rate, D_rate))

def FeatureExtraction_NCM_NCA_Dataset(dataset_path: str, Tem: int, C_rate: int, D_rate: int, TPidx: float, PSPidx: float, InitialCapacity: int):
    files= os.listdir(dataset_path)
    feature_name_ls= ["mean", "std", "var", "skew", "kurt", "rms", "max", "min", "median", "p2p"]
    df_res= pd.DataFrame(columns= ['Cell', 'Cycle', 'Voltages', 'Rate', 'Tem', 'Capacity'])
    count= 0
    rate= f"{C_rate}" if C_rate== 1 else f"0{C_rate}"
    charging_policy= f"CY{Tem}-{rate}_{D_rate}"
    feature_elaxation= pd.DataFrame()
    for file_index, file in enumerate(files):
        if charging_policy in file:
            max_relexation_voltage, min_relexation_voltage= -np.inf, np.inf
            data_r= pd.read_csv(os.path.join("battery-data-nca-ncm-cells", file))
            k= np.min(data_r['cycle number'].values)
            data_k= data_r[data_r['cycle number'] == k]
            Q_p = np.max(np.array(data_k['Q discharge/mA.h']))
            delta = 1
            for cycle in range(int(np.min(data_r['cycle number'].values)), int(np.max(data_r['cycle number'].values))+1):
                data_i= data_r[data_r['cycle number'] == cycle]
                Ecell = np.array(data_i['Ecell/V'])
                Q_dis = np.array(data_i['Q discharge/mA.h'])
                Current = np.array(data_i['<I>/mA'])
                Control = np.array(data_i['control/V/mA'])
                Cr = np.array(data_i['control/mA'])[1]/ 2500
                Cr_d = int(files[file_index][8])
                if np.max(Q_dis) < 1650 or np.max(Q_dis) > 2510:
                    delta = delta + 1
                    continue
                if np.abs(np.max(Q_dis) - Q_p) > delta * 10:
                    delta = delta + 1
                    continue
                delta= 1
                Q_p = np.max(Q_dis)
                index = np.where(np.abs(Control) == 0)
                if index[0][0] > 0:
                    start = index[0][0]
                else:
                    start = index[0][14]
                if Control[start + 19] == 0 and np.max(Q_dis)< InitialCapacity* TPidx and np.max(Q_dis)> InitialCapacity* PSPidx:
                # if Control[start + 19] == 0 :
                    count+= 1
                    relaxation_voltage= pd.DataFrame({"Voltage": Ecell[start: start+ 14]})
                    if max_relexation_voltage< np.max(relaxation_voltage): 
                        max_relexation_voltage= np.max(relaxation_voltage)
                    if min_relexation_voltage> np.min(relaxation_voltage): 
                        min_relexation_voltage= np.min(relaxation_voltage)
                    MEAN, STD, VAR, SKEW, KURT, RMS= relaxation_voltage.mean(), relaxation_voltage.std(), relaxation_voltage.var(), relaxation_voltage.skew(), relaxation_voltage.kurt(), (relaxation_voltage**2).mean()** (1/2)
                    MAX, MIN, MEDIAN, PEAK2PEAK= relaxation_voltage.max(), relaxation_voltage.min(), relaxation_voltage.median(), relaxation_voltage.max()- relaxation_voltage.min()
                    feature_elaxation_voltage= pd.concat([MEAN, STD, VAR, SKEW, KURT, RMS, MAX, MIN, MEDIAN, PEAK2PEAK])
                    feature_elaxation_voltage.index= np.array([["V-"+ f_name] for f_name in feature_name_ls]).ravel()
                    feature_elaxation_voltage= feature_elaxation_voltage.to_frame().T
                    feature_elaxation_voltage["Charging-Policy"]= [charging_policy]
                    feature_elaxation_voltage["Cell-Name"]= [charging_policy+ "-"+ file.split(".")[0].split("-")[-1]]
                    feature_elaxation_voltage["Cycle"]= [cycle]
                    feature_elaxation= pd.concat([feature_elaxation, feature_elaxation_voltage], axis= 0, ignore_index= 0)
                    df_res= pd.concat([df_res, pd.DataFrame({'Cell': file.split("#")[-1].split(".")[0], 
                                                                                          'Cycle': cycle, 
                                                                                          'Voltages': Ecell[start: start+ 14], 
                                                                                          'Rate': Cr, 
                                                                                          'Tem': Tem, 
                                                                                          'Capacity': np.max(Q_dis)})], 
                                                                                          axis= 0, 
                                                                                          ignore_index= True)
            feature_elaxation= feature_elaxation.reset_index(drop= True)
            feature_elaxation.to_csv(os.path.join(os.getcwd(), "data", "feature-elaxation-nca-ncm-tem%s-Crate%s-Drate%s.csv"%(Tem, C_rate, D_rate)), index= False)
    # print("NCM & NCA in Tem: %s, C-rate: %s, D-rate: %s is finished !!"%(Tem, C_rate, D_rate))

if __name__== "__main__":
    time_interval_ratio= 0.08
    sliding_step= 0.01
    for start_ratio in np.arange(0.0, 1.0, sliding_step):
        TPidx= round(1- start_ratio, 4)
        PSPidx, InitialCapacity= round(TPidx- time_interval_ratio, 4), 3500
        if PSPidx< 0.80:
            break
        print("Processing time interval: from %.4f to %.4f"%(TPidx, PSPidx))
        nca_ls= [(25, 5, 1), (25, 5, 1), (25, 25, 1), (35, 5, 1), (45, 5, 1)]
        for Tem, C_rate, D_rate in nca_ls:
            FeatureExtraction_NCA_Dataset(dataset_path= "battery-data-nca-cells", Tem= Tem, C_rate= C_rate, D_rate= D_rate, TPidx= TPidx, PSPidx= PSPidx, InitialCapacity= InitialCapacity)
        ncm_ls= [(25, 5, 1), (35, 5, 1), (45, 5, 1)]
        for Tem, C_rate, D_rate in ncm_ls:
            FeatureExtraction_NCM_Dataset(dataset_path= "battery-data-ncm-cells", Tem= Tem, C_rate= C_rate, D_rate= D_rate, TPidx= TPidx, PSPidx= PSPidx, InitialCapacity= InitialCapacity)
        InitialCapacity= 2500
        nca_ncm_ls= [(25, 5, 1), (25, 5, 2), (25, 5, 4)]
        for Tem, C_rate, D_rate in nca_ncm_ls:
            FeatureExtraction_NCM_NCA_Dataset(dataset_path= "battery-data-nca-ncm-cells", Tem= Tem, C_rate= C_rate, D_rate= D_rate, TPidx= TPidx, PSPidx= PSPidx, InitialCapacity= InitialCapacity)
        domain_feature_dataset= ReadDatasetsAndMergeDatasets("data")
        scaler= StandardScaler()
        label= domain_feature_dataset["Charging-Policy"].to_numpy()
        domain_feature_dataset= domain_feature_dataset.drop(["Charging-Policy", "Cell-Name", "Cycle"], axis= 1)
        traindata_feature= scaler.fit_transform(domain_feature_dataset)
        ### T-SNE method
        X_embedded= TSNE(n_components= 2, learning_rate= 'auto', init= "pca").fit_transform(traindata_feature)
        plt.figure(figsize= (23, 23))
        for cell_idx in np.unique(label):
            mask= label== cell_idx
            if len(cell_idx.split(" ")[0])> 3:
                cell_idx= cell_idx[: 3]+ " & "+ cell_idx[3: ]
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],  label= f"{cell_idx}",  alpha= 0.7, edgecolors= "black", s= 400, linewidths= 1.2)
        plt.legend(fontsize= 31, loc= "lower center", bbox_to_anchor= (0.49, 1.00), ncol= 4, frameon= False)
        plt.xlabel("Component#1", fontsize= 45)
        plt.ylabel("Component#2", fontsize= 45)
        plt.xticks(fontsize= 45)
        plt.yticks(fontsize= 45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("fig", "feature-distribution-%.4f-%.4f.png")%(TPidx, PSPidx), dpi= 600)
        plt.show()