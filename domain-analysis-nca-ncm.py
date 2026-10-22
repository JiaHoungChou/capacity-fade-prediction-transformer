import os
import numpy as np
import pandas as pd
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

if __name__== "__main__":
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
    plt.savefig(os.path.join("fig", "feature-distribution.png"), dpi= 600)
    plt.show()