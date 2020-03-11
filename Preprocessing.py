import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
import tqdm
import pickle
import csv
import math
import warnings
import json
warnings.filterwarnings("ignore")

def main(config):    
    pool = mp.Pool(processes=config["n_workers"])
    pool.map(job, range(1, 76))
    
# Return one hot encoding of length = base and the nth element = 1
def make_onehot(n, base):
    if n == -1 :
        return list(np.zeros(base))
    onehot = np.zeros(base)
    onehot[n-1] = 1
    return list(onehot)

# Return the time slot number of the given datatime boject
def is_time_slot(datetime, time_slot_comp):
    slot = 0
    for i in range(4):
        if datetime.time() >= time_slot_comp[i][0] and datetime.time() < time_slot_comp[i][1]:
            slot = i
    return slot

# Return [week, time slot, time slot of the week]
def time_slot(datetime, begin_datetime, end_datetime, time_slot_comp):
    if datetime < begin_datetime or datetime > end_datetime:
        return [-1, -1, -1]
    diff = datetime - begin_datetime
    time_slot = is_time_slot(datetime, time_slot_comp)
    return [int(diff.days / 7), diff.days*4 + time_slot, datetime.weekday()*4 + time_slot]

# Return the scaleed played duration value
def scale_played_duration(n):
    return math.log(1 + n / 9550675.0)

# Return the Processed data
def prepare_data(row, platform):
    temp = list()
    
    # platform
    temp += make_onehot(platform[row[4]], 3)

    # connection type
    if 'wifi' in row[7]:
        connect = make_onehot(0, 3)
    elif 'cellular' in row[7]:
        connect = make_onehot(1, 3)
    elif 'online' in row[7]:
        connect = make_onehot(2, 3)
    else:
        connect = make_onehot(-1, 3)
    temp += connect

    # watch ratio
    temp.append(float(row[5]) / (float(row[6]) + 1e-10))
    
    # total number of episode
    temp.append(math.log(1 + float(row[5]) / 210.0))

    # limit playzone countdown
    temp.append(1 if 'limit playzone countdown' in row[3] else 0)

    # error    
    temp.append(1 if 'error' in row[3] else 0)
    
    # (video ended, program stopped or enlarged-reduced, program stopped)
    temp.append(1 if 'ed' in row[3] else 0)  
    
    # played duration of popular title
    temp.append(math.log(1 + float(row[2]) / 5224.0) if int(row[0]) in [74, 79, 77]  else 0)
    
    return temp

# Extact selected features from data files(This part could be a bit messy)
def job(k):
    with open("preprocessing_config.json") as f:
        config = json.load(f)
    begin_datetime = datetime.strptime('2017-01-02 01:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
    end_datetime = datetime.strptime('2017-08-14 01:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
    time_slot_comp = [[datetime.strptime('01:00:00', '%H:%M:%S').time(), datetime.strptime('09:00:00', '%H:%M:%S').time()], 
                 [datetime.strptime('09:00:0', '%H:%M:%S').time(), datetime.strptime('17:00:00', '%H:%M:%S').time()], 
                 [datetime.strptime('17:00:0', '%H:%M:%S').time(), datetime.strptime('21:00:00', '%H:%M:%S').time()], 
                 [datetime.strptime('21:00:00', '%H:%M:%S').time(), datetime.strptime('01:00:00', '%H:%M:%S').time()]]
    platform = { 'Web': 0, 'iOS': 1, 'Android': 2 }
    
    file = pd.read_csv((config["data_folder"] + "/data-0{:0>2d}.csv").format(k))
    n = file.groupby(['user_id']).size().values
    file = file.drop(columns=['user_id', 'device_id', 'session_id', 'is_trailer'])
    file = np.split(np.asarray(file.values), np.add.accumulate(n), axis=0)[:len(n)]
    
    pad = config["pad_token"]
    collect = []
    for idx in file:
        person_data = np.ones((32, 28, 13)) * pad
        week_data = np.ones((28, 13)) * pad
        time_slot_data = []
        watch_time_sum = 0
        prev = time_slot(datetime.strptime(idx[0][1], '%Y-%m-%d %H:%M:%S.%f'), begin_datetime, end_datetime, time_slot_comp)
        for row in idx:
            now = time_slot(datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f'), begin_datetime, end_datetime, time_slot_comp)
            if prev[0] == -1:
                prev = now
            if now[0] == -1:
                continue
            if now[1] == prev[1]:
                time_slot_data.append(prepare_data(row, platform))
                watch_time_sum += float(row[2])
            else:
                time_slot_data = np.mean(time_slot_data, axis=0).tolist()
                time_slot_data.append(scale_played_duration(watch_time_sum))
                week_data[prev[2]] = time_slot_data
                time_slot_data = []
                
                if prev[0] < now[0]:
                    person_data[prev[0]] = week_data
                    week_data  = week_data * 0 + pad

                time_slot_data.append(prepare_data(row, platform))            
                watch_time_sum += float(row[2])
            prev = now
                
        if len(time_slot_data) != 0:            
            time_slot_data = np.mean(time_slot_data, axis=0).tolist()
            time_slot_data.append(scale_played_duration(watch_time_sum))
            week_data[prev[2]] = time_slot_data
            person_data[prev[0]] = week_data            
        collect.append(person_data)
        
    with open((config["preprocessed_data_folder"] +"/data-00{:0>2d}_preprocessed.pkl").format(k), 'wb') as handle:
        pickle.dump(collect, handle)
        
if __name__ == '__main__':
    with open("preprocessing_config.json") as f:
        config = json.load(f)
    main(config)