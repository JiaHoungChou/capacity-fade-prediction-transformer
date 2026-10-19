import os, torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from utilize.transformer import Transformer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
plt.rcParams["font.family"] = "Times New Roman"

def SlidingWindowMultiStep(x: pd.DataFrame, window_size: int, n_step_ahead_size: int):
    encoder_x, decoder_x, decoder_y, decoder_capacity_y= [], [], [], []
    for idx in range(len(x)- window_size- n_step_ahead_size+ 1):
        encoder_window_x= x[idx: idx+ window_size]
        last_y= x[idx+ window_size- 1, -1]
        previous_y= np.concatenate([[last_y], x[idx+ window_size: idx+ window_size+ n_step_ahead_size- 1, -1]])
        future_y= x[idx+ window_size: idx+ window_size+ n_step_ahead_size, -1]
        future_fade= np.maximum(previous_y- future_y, 0.0)
        historical_fade= max(x[idx+ window_size- 2, -1]- last_y, 0.0)
        decoder_window_x= np.concatenate([[historical_fade], future_fade[: -1]])
        encoder_x.append(encoder_window_x)
        decoder_x.append(decoder_window_x[:, None])
        decoder_y.append(future_fade[:, None])
        decoder_capacity_y.append(future_y[:, None])
    return (np.array(encoder_x), np.array(decoder_x), np.array(decoder_y), np.array(decoder_capacity_y))

def PositiveCapacityFade(x):
    return nn.functional.softplus(x)+ 1e-6

def AutoRegressiveValidation(model, encoder_x, n_step_ahead_size: int, fade_scale: float, device= "cuda", max_decoder_context= 100):
    if not isinstance(encoder_x, torch.Tensor):
        encoder_x= torch.FloatTensor(encoder_x).to(device)
    else:
        encoder_x= encoder_x.to(device)
    ### last capacity fade
    historical_fade= torch.relu(encoder_x[:, -2: -1, -1:]- encoder_x[:, -1:, -1:])
    decoder_input= historical_fade/ fade_scale
    rolling_fade_prediction= []
    model.eval()
    with torch.inference_mode():
        for time_step in range(n_step_ahead_size):
            output= model(encoder_x, decoder_input)
            output= PositiveCapacityFade(output[:, -1:, :])
            rolling_fade_prediction.append(output)
            if time_step== n_step_ahead_size- 1:
                continue
            decoder_input= torch.cat([decoder_input, output], dim= 1)
            if decoder_input.size(1)> max_decoder_context:
                decoder_input= decoder_input[:, -max_decoder_context:, :]
    rolling_fade_prediction= torch.cat(rolling_fade_prediction, dim= 1)
    rolling_capacity_prediction= encoder_x[:, -1: , -1: ]- torch.cumsum(rolling_fade_prediction* fade_scale, dim= 1)
    return rolling_capacity_prediction.detach().cpu().numpy()

def AutoRegressivePrediction(model, encoder_x, n_step_ahead_size, eol_threshold, fade_scale: float, device= "cuda", max_decoder_context= 100):
    if not isinstance(encoder_x, torch.Tensor):
        encoder_x= torch.FloatTensor(encoder_x)
    encoder_x= encoder_x.to(device)
    assert encoder_x.size(0)== 1, "AutoRegressivePrediction is designed for one prediction starting point."
    historical_fade= torch.relu(encoder_x[:, -2:-1, -1:]- encoder_x[:, -1:, -1:])
    decoder_input= historical_fade/ fade_scale
    current_capacity= encoder_x[:, -1:, -1:]
    rolling_capacity_prediction= []
    model.eval()
    with torch.inference_mode():
        for time_step in range(n_step_ahead_size):
            output= model(encoder_x, decoder_input)
            predicted_fade= PositiveCapacityFade(output[:, -1:, :])
            current_capacity= current_capacity- predicted_fade* fade_scale
            rolling_capacity_prediction.append(current_capacity.cpu())
            if current_capacity.item()<= float(np.asarray(eol_threshold).reshape(-1)[0]):
                print("Step-ahead prediction reached EOL threshold at step: ", time_step+ 1)
                break
            decoder_input= torch.cat([decoder_input, predicted_fade], dim= 1)
            if decoder_input.size(1)> max_decoder_context:
                decoder_input= decoder_input[:, -max_decoder_context:, :]
    rolling_capacity_prediction= torch.cat(rolling_capacity_prediction, dim= 1)
    return rolling_capacity_prediction.numpy()

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    print("computing device is using: ", device)
    with open(os.path.join(os.getcwd(), "ncm.csv"), "r", encoding= "utf-8") as file:
        data= pd.read_csv(file)
    ###############################################################
    plt.figure(figsize= (8, 8))
    plt.subplot(2, 1, 1)
    plt.title("Voltage Pattern in Each Cycling", fontsize= 22)
    plt.scatter(np.arange(0, len(data[data["Cell"]== 1]["Voltages"])), data[data["Cell"]== 1]["Voltages"], edgecolors= "black", color= "darkblue")
    plt.plot(np.arange(0, len(data[data["Cell"]== 1]["Voltages"])), data[data["Cell"]== 1]["Voltages"], color= "darkblue")
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.title("Capacity Trend", fontsize= 22)
    plt.scatter(np.arange(0, len(data[data["Cell"]== 1]["Capacity"])), data[data["Cell"]== 1]["Capacity"], edgecolors= "black", color= "darkblue")
    plt.plot(np.arange(0, len(data[data["Cell"]== 1]["Capacity"])), data[data["Cell"]== 1]["Capacity"], color= "darkblue")
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    ###############################################################
    known_traindata= data[data["Cell"]== 1]
    TPidx= 0.90
    PSPidx= 0.86
    InitialCapacity= 3500
    known_traindata= known_traindata[known_traindata["Capacity"]> InitialCapacity* PSPidx]
    known_traindata= known_traindata[known_traindata["Capacity"]< InitialCapacity* TPidx]
    plt.figure(figsize= (8, 4))
    plt.title("Known Capacity Trend - PSP split", fontsize= 25)
    plt.plot(np.arange(known_traindata["Capacity"].index[0], known_traindata["Capacity"].index[-1]+ 1), known_traindata["Capacity"], color= "darkblue", label= "Known Capacity")
    plt.plot(np.arange(known_traindata["Capacity"].index[-1], len(data[data["Cell"]== 1]["Capacity"])), data[data["Cell"]== 1]["Capacity"][known_traindata["Capacity"].index[-1]: ], color= "darkred", ls= "--", label= "Unknown Capacity")
    plt.scatter(known_traindata["Capacity"].index[-1], known_traindata["Capacity"].iloc[-1], color= "black", edgecolors= "darkred", s= 100, marker= "x", label= "Prediction Start Point (PSP)")
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.grid(True)
    plt.legend(fontsize= 15)
    plt.show()
    known_traindata= known_traindata.to_numpy()
    scaler, scaler_y= StandardScaler(), StandardScaler()
    scaler_y.fit(known_traindata[:, -1:])
    known_traindata= scaler.fit_transform(known_traindata)
    ###############################################################
    print("shape of known_traindata: ", known_traindata.shape)
    sliding_window_size= 128
    n_step_ahead_size= 400
    encoder_x, decoder_x, decoder_y, decoder_capacity_y= SlidingWindowMultiStep(known_traindata, sliding_window_size, n_step_ahead_size)
    positive_fade= decoder_y[decoder_y> 0]
    fade_scale= max(float(np.median(positive_fade)), 1e-6) if len(positive_fade)> 0 else 1.0
    decoder_x= decoder_x/ fade_scale
    decoder_y= decoder_y/ fade_scale
    print("capacity fade scale: ", fade_scale)
    print("shape of encoder_x: ", encoder_x.shape)
    print("shape of decoder_x: ", decoder_x.shape)
    print("shape of decoder_y: ", decoder_y.shape)
    print("shape of decoder_capacity_y: ", decoder_capacity_y.shape)
    ###############################################################
    batch_size= 32
    epoch= 100
    lr= 0.0008
    autoregressive_finetune_ratio= 0.20
    autoregressive_finetune_start= int(epoch* (1.0- autoregressive_finetune_ratio))
    max_decoder_context= 500
    validation_interval= 5
    validation_sample_size= min(5, encoder_x.shape[0])
    src_feature_size= encoder_x.shape[-1]
    trg_feature_size= decoder_x.shape[-1]
    model= Transformer(src_feature_size, trg_feature_size, embed_size= 64, heads= 4, num_layers= 2, max_length= n_step_ahead_size).to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr= lr, weight_decay= 1e-4)
    MSELossFunction= nn.MSELoss()
    ###############################################################
    encoder_x= torch.FloatTensor(encoder_x)
    decoder_x= torch.FloatTensor(decoder_x)
    decoder_y= torch.FloatTensor(decoder_y)
    decoder_capacity_y= torch.FloatTensor(decoder_capacity_y)
    validation_encoder_x= encoder_x[- validation_sample_size: ]
    validation_capacity_y= decoder_capacity_y[- validation_sample_size: ]
    TensorLoaderDataSet= TensorDataset(encoder_x, decoder_x, decoder_y, decoder_capacity_y)
    TensorLoaderDataSet= DataLoader(TensorLoaderDataSet, batch_size, shuffle= True, pin_memory= device== "cuda")
    validation_mse_loss= np.inf
    for idx in range(epoch):
        model.train()
        training_mode= "Parallel" if idx< autoregressive_finetune_start else "Autoregressive"
        epoch_total_loss= torch.tensor(0.0, device= device)
        epoch_fade_loss= torch.tensor(0.0, device= device)
        epoch_capacity_loss= torch.tensor(0.0, device= device)
        ### epoch_endpoint_loss: 
        epoch_endpoint_loss= torch.tensor(0.0, device= device)
        total_samples= 0
        for encoder_batch_x, decoder_batch_x, decoder_batch_y, decoder_batch_capacity_y in TensorLoaderDataSet:
            encoder_batch_x= encoder_batch_x.to(device, non_blocking= True)
            decoder_batch_x= decoder_batch_x.to(device, non_blocking= True)
            decoder_batch_y= decoder_batch_y.to(device, non_blocking= True)
            decoder_batch_capacity_y= decoder_batch_capacity_y.to(device, non_blocking= True)
            current_batch_size= encoder_batch_x.size(0)
            optimizer.zero_grad()
            if idx< autoregressive_finetune_start:
                ### Fast parallel teacher-forcing training. Causal mask prevents the decoder from seeing future steps.
                output= model(encoder_batch_x, decoder_batch_x)
                rolling_fade_prediction= PositiveCapacityFade(output)
            else:
                ### Autoregressive fine-tuning is only used in the last training stage to reduce exposure bias.
                decoder_input= decoder_batch_x[:, : 1, :]
                rolling_fade_prediction= []
                for time_step in range(decoder_batch_y.size(1)):
                    output= model(encoder_batch_x, decoder_input)
                    output= PositiveCapacityFade(output[: , -1: , : ])
                    rolling_fade_prediction.append(output)
                    if time_step== decoder_batch_y.size(1)- 1:
                        continue
                    decoder_input= torch.cat([decoder_input, output.detach()], dim= 1)
                    if decoder_input.size(1)> max_decoder_context:
                        decoder_input= decoder_input[:, -max_decoder_context: , :]
                rolling_fade_prediction= torch.cat(rolling_fade_prediction, dim= 1)
            rolling_capacity_prediction= encoder_batch_x[:, -1:, -1:]- torch.cumsum(rolling_fade_prediction* fade_scale, dim= 1)
            fade_loss= MSELossFunction(rolling_fade_prediction, decoder_batch_y)
            capacity_loss= MSELossFunction(rolling_capacity_prediction, decoder_batch_capacity_y)
            endpoint_loss= MSELossFunction(rolling_capacity_prediction[:, -1:, :], decoder_batch_capacity_y[:, -1:, :])
            loss= fade_loss+ 0.2* capacity_loss+ 0.2* endpoint_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
            optimizer.step()
            epoch_fade_loss+= fade_loss.detach()* current_batch_size
            epoch_capacity_loss+= capacity_loss.detach()* current_batch_size
            epoch_endpoint_loss+= endpoint_loss.detach()* current_batch_size
            epoch_total_loss+= loss.detach()* current_batch_size
            total_samples+= current_batch_size
        avg_total_loss= (epoch_total_loss/ total_samples).item()
        avg_fade_loss= (epoch_fade_loss/ total_samples).item()
        avg_capacity_loss= (epoch_capacity_loss/ total_samples).item()
        avg_endpoint_loss= (epoch_endpoint_loss/ total_samples).item()
        print("epoch: %03d// training mode: %s// total loss: %02.4f// fade loss: %02.4f// capacity loss: %02.4f// endpoint loss: %02.4f"%(idx+ 1, training_mode, avg_total_loss, avg_fade_loss, avg_capacity_loss, avg_endpoint_loss))
        if (idx+ 1)% validation_interval== 0:
            auto_regressive_prediction= AutoRegressiveValidation(model, validation_encoder_x, n_step_ahead_size= validation_capacity_y.size(1), fade_scale= fade_scale, device= device, max_decoder_context= max_decoder_context)
            auto_regressive_prediction= auto_regressive_prediction[:, :, -1]
            valid_mse= ((auto_regressive_prediction- validation_capacity_y[:, :, -1].detach().cpu().numpy())** 2).mean()
            print("validation loss at epoch %03d// autoregressive capacity mse loss: %02.4f"%(idx+ 1, valid_mse))
            if valid_mse< validation_mse_loss:
                validation_mse_loss= valid_mse
                torch.save(model.state_dict(), "best_transformer_model.pth")
                print("Best model saved at epoch %03d with validation loss: %02.4f"%(idx+ 1, valid_mse))
            if (idx+ 1)% 10== 0:
                fig, axes= plt.subplots(validation_sample_size, 1, figsize= (8, 3* validation_sample_size))
                axes= np.atleast_1d(axes)
                fig.suptitle("Validation - Autoregressive Capacity Prediction", fontsize= 18)
                for i in range(validation_sample_size):
                    ax= axes[i]
                    ax.plot(np.arange(n_step_ahead_size), auto_regressive_prediction[-(i+ 1)], color= "darkred", label= "Prediction")
                    ax.plot(np.arange(n_step_ahead_size), validation_capacity_y[-(i+ 1), :, -1].detach().cpu().numpy(), color= "darkblue", label= "Ground Truth")
                    cycle_index= np.arange(encoder_x.shape[0]- i, encoder_x.shape[0]- i+ n_step_ahead_size)
                    tick_interval= 10
                    x= np.arange(n_step_ahead_size)
                    ax.set_xticks(x[::tick_interval])
                    ax.set_xticklabels(cycle_index[::tick_interval], fontsize= 11, rotation= 90)
                    ax.tick_params(axis= "y", labelsize= 11)
                    ax.set_xlabel("Cycle Index", fontsize= 13)
                    ax.set_ylabel("Standardized Capacity", fontsize= 13)
                    ax.grid(True)
                    ax.legend(fontsize= 11)
                plt.tight_layout()
                plt.show()
    ################################################################
    ### rolling prediction for RUL prediction until EOL.
    model.load_state_dict(torch.load("best_transformer_model.pth", map_location= device))
    encoder_psp= torch.FloatTensor(known_traindata[-sliding_window_size:][None, :, :])
    eol_threshold= scaler_y.transform(np.array([[0.80* InitialCapacity]]))
    auto_regressive_prediction= AutoRegressivePrediction(model, encoder_psp, n_step_ahead_size= 10000, eol_threshold= eol_threshold, fade_scale= fade_scale, device= device, max_decoder_context= max_decoder_context)
    auto_regressive_prediction= auto_regressive_prediction[:, :, -1].reshape(-1)
    auto_regressive_prediction= scaler_y.inverse_transform(auto_regressive_prediction.reshape(-1, 1)).reshape(-1)
    unknown_testdata= data[data["Cell"]== 1]
    unknown_testdata= unknown_testdata[unknown_testdata["Capacity"]< InitialCapacity* PSPidx]
    unknown_testdata= unknown_testdata[unknown_testdata["Capacity"]> InitialCapacity* 0.80]
    known_traindata= data[data["Cell"]== 1]
    known_traindata= known_traindata[known_traindata["Capacity"]> InitialCapacity* PSPidx]
    plt.figure(figsize= (10, 6))
    plt.title("Unknown Capacity Prediction Trend", fontsize= 25)
    plt.plot(np.arange(known_traindata["Capacity"].index[0], known_traindata["Capacity"].index[-1]+ 1), known_traindata["Capacity"], color= "darkblue", label= "Known Capacity")
    plt.plot(np.arange(known_traindata["Capacity"].index[-1], len(data[data["Cell"]== 1]["Capacity"])), data[data["Cell"]== 1]["Capacity"][known_traindata["Capacity"].index[-1]: ], color= "darkred", ls= "--", label= "Unknown Capacity")
    plt.scatter(known_traindata["Capacity"].index[-1], known_traindata["Capacity"].iloc[-1], color= "black", s= 100, marker= "x", label= "Prediction Start Point (PSP)")
    plt.plot(np.arange(unknown_testdata.index[0], unknown_testdata.index[-1]+ 1), unknown_testdata["Capacity"], color= "darkblue", label= "Unknown Capacity")
    plt.plot(np.arange(unknown_testdata.index[0], unknown_testdata.index[0]+ len(auto_regressive_prediction)), auto_regressive_prediction, color= "darkgreen", label= "Unknown Predicted Capacity")
    plt.hlines(y= 0.80* InitialCapacity, xmin= 0, xmax= len(data[data["Cell"]== 1]["Capacity"]), color= "darkorange", ls= "--", label= "EOL Threshold (80% of Nominal Capacity)")
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.grid(True)
    plt.legend(fontsize= 15)
    plt.show()