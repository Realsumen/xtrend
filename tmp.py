import tensorflow as tf
import importlib
import os
import loss_functions
import dataprocessor
import model
import pickle
import change_point_detection
from tqdm import tqdm
from heartrate import trace

def reload_custom_libs():
    importlib.reload(loss_functions)
    importlib.reload(dataprocessor)
    importlib.reload(change_point_detection)
    importlib.reload(model)
reload_custom_libs()
from change_point_detection import *
from loss_functions import *
from model import *
from dataprocessor import *

macd_timescales = [(8, 24), (16, 28), (32, 96)]
rtn_timescales = [1, 21, 63, 126, 252]
timesteps = 126
folder_path = "data"
files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
异常数据元组 = ("CC00.NYB.xlsx", "LB00.CME.xlsx", "ES00.CME.xlsx", "NQ00.CME.xlsx", "YM00.CBT.xlsx", "SP00.CME.xlsx")
files = [file for file in files if file not in 异常数据元组]



# 加载文件
# data_list = process_data_list(files, macd_timescales, rtn_timescales, test=200)
data_list = process_data_list(files, macd_timescales, rtn_timescales)
# 获得断点分割片段数据
gaussion_process_list = get_segment_list(data_list=data_list)

with open('all.pkl', 'wb') as f:
    pickle.dump(gaussion_process_list, f)
with open('all.pkl', 'rb') as f:
    gaussion_process_list = pickle.load(f)
    