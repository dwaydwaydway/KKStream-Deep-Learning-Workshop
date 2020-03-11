from Model import Model
from Dataset import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")

def main(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    for i in tqdm(range(46,76)):
        if i == 46 :
            test_df = pd.read_pickle((config["preprocessed_data_folder"] + "/data-00{:0>2d}_preprocessed.pkl").format(i))
        else:
            test_df = np.concatenate((test_df, pd.read_pickle((config["preprocessed_data_folder"] + "/data-00{:0>2d}_preprocessed.pkl").format(i))), axis=0)
    
    model = Model(32, 28, 13, 0.3, device=device).to(device)
    model.load_state_dict(torch.load(config["model_file"]))
    model.eval()

    testing_set = Dataset(test_df, training=False)
    testing_generator = data.DataLoader(testing_set, shuffle=False, **config["dataloader_params"])

    test = []
    i = 0
    with torch.set_grad_enabled(False):
        for local_batch in tqdm(testing_generator):
            # Transfer to GPU
            local_batch = local_batch.type(torch.FloatTensor).to(device)
            outputs = []
            outputs.append(torch.sigmoid(model(local_batch)).cpu().numpy())

            test = test + list(sum(outputs) / len(outputs))

    columns = ['time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3', 'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7', 'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11', 'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15', 'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19', 'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23', 'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27']
    
    predict_df = pd.DataFrame(test, columns= columns)
    predict_df.insert(loc=0, column='user_id', value=range(57159, 57159 + 37092))
    predict_df.to_csv("submission.csv", index=False)
    
if __name__ == '__main__':
    with open("testing_config.json") as f:
        config = json.load(f)
    main(config)