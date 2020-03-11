import numpy as np
from tqdm import tqdm
import pickle
import math
import pandas as pd
from Loss_functions import FocalLoss
from Model import Model
from Dataset import Dataset
import torch
import torch.nn.functional as F
import json
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def main(config):
    label_df = []
    print("loading data.....")
    for i in tqdm(range(1, 46)):
        if i == 1:
            train_df = pd.read_pickle((config["preprocessed_data_folder"] + "/data-00{:0>2d}_preprocessed.pkl").format(i))
            label_df.append(pd.read_csv((config["label_folder"] + "/label-0{:0>2d}.csv").format(i)))
        else:
            train_df = np.concatenate((train_df, pd.read_pickle((config["preprocessed_data_folder"] + "/data-00{:0>2d}_preprocessed.pkl").format(i))), axis=0)
            label_df.append(pd.read_csv((config["label_folder"] + "/label-0{:0>2d}.csv").format(i)))

    label_df = pd.concat(label_df)
    label_df = label_df.drop(columns=['user_id'])
    time_slot = [x for x in label_df.columns if x.startswith("time_slot")]
    label_df = label_df.values
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_df, label_df, test_size=0.1, random_state=42)
    
    # Generators
    training_set = Dataset(X_train, y_train)
    training_generator = data.DataLoader(training_set, shuffle=True, **config["dataloader_params"])

    validation_set = Dataset(X_valid, y_valid)
    validation_generator = data.DataLoader(validation_set, shuffle=True, **config["dataloader_params"])

    model = Model(32, 28, 13, 0.35, device=device).to(device)

    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    for epoch in range(config["max_epochs"]):
        # Training
        print("\nepoch: {}".format(epoch+1))
        train_loss = [0, 0, 0]
        valid_loss = [0, 0, 0]
        step = [0, 0]
        for local_batch, local_labels in tqdm(training_generator):
            # Transfer to GPU
            local_batch, local_labels = local_batch.type(torch.FloatTensor).to(device), local_labels.type(torch.FloatTensor).to(device)
            outputs = model(local_batch.to(device))
            loss =  criterion(outputs, local_labels)#f1_loss(outputs, local_labels)*0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            local_labels = local_labels.transpose(0, 1).cpu().numpy()
            outputs = outputs.transpose(0, 1).cpu().detach().numpy()

            roc = 0
            d_roc = 0
            f1 = 0
            discard_sample = 0
            for i in range(28):
                flag = 0
                try:
                    roc += roc_auc_score(local_labels[i], outputs[i])
                    d_roc += roc_auc_score(local_labels[i], outputs[i] > 0.5)
                    f1 += f1_score(local_labels[i], outputs[i] > 0.5, average='macro')  
                except ValueError:
                    discard_sample += 1
            train_loss[0] = (roc/(28 - discard_sample) + train_loss[0])
            train_loss[1] = (d_roc/(28 - discard_sample) + train_loss[1])
            train_loss[2] = (f1/28 + train_loss[2])
            step[0] += 1
            tqdm.write("Train AUROC: {:.4f}, d-AUROC: {:.4f}, F1: {:.4f}".format(train_loss[0] / step[0], train_loss[1] / step[0], train_loss[2] / step[0]), end='\r')
        
        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in tqdm(validation_generator):
                # Transfer to GPU
                local_batch, local_labels = local_batch.type(torch.FloatTensor).to(device), local_labels.type(torch.FloatTensor).to(device)
                outputs = model(local_batch)

                local_labels = local_labels.transpose(0, 1).cpu().numpy()
                outputs = outputs.transpose(0, 1).cpu().detach().numpy()

                roc = 0
                d_roc = 0
                f1 = 0
                discard_sample = 0
                for i in range(28):
                    try:
                        roc += roc_auc_score(local_labels[i], outputs[i])
                        d_roc += roc_auc_score(local_labels[i], outputs[i] > 0.5)
                        f1 += f1_score(local_labels[i], outputs[i] > 0.5, average='macro')  
                    except ValueError:
                        discard_sample += 1
                valid_loss[0] = (roc/(28 - discard_sample) + valid_loss[0])
                valid_loss[1] = (d_roc/(28 - discard_sample) + valid_loss[1])
                valid_loss[2] = (f1/28 + valid_loss[2])
                step[1] += 1
                tqdm.write("Valid AUROC: {:.4f}, d-AUROC: {:.4f}, F1: {:.4f}".format(valid_loss[0] / step[1], valid_loss[1] / step[1], valid_loss[2] / step[1]), end='\r')
                
        torch.save(model.state_dict(), (config["model_folder"] + "/model_{}.pkl").format(epoch+1))
        
if __name__ == '__main__':
    with open("training_config.json") as f:
        config = json.load(f)
    main(config)