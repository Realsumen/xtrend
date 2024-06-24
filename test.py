import heartrate
import importlib
import os
from models import *
from dataprocessor import *
def reload_custom_libs():
    import models
    import dataprocessor
    importlib.reload(models)
    importlib.reload(dataprocessor)

heartrate.trace(browser=True)
macd_timescales = [(8, 24), (16, 28), (32, 96)]
rtn_timescales = [1, 21, 63, 126, 252]
timesteps = 16

folder_path = 'data'
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
files = [file for file in files if file not in ("CC00.NYB.xlsx", "LB00.CME.xlsx", "ES00.CME.xlsx", "NQ00.CME.xlsx", "YM00.CBT.xlsx")]
data_list = []


data_list = process_data_list(files, macd_timescales, rtn_timescales, test = True)
target_set, labels, map = generate_tensors(data_list, timesteps, encoder_type = "one-hot", return_map=True)
context_set, _ = generate_tensors(data_list, timesteps, encoder_type = "one-hot", contain_next_day_rtn=True)

data_binder(context_set, target_set, labels)