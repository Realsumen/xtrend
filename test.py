import tensorflow as tf
import importlib
import os
import pickle
from models import *
from dataprocessor import *
from tqdm import tqdm
from change_point_detection import *
def reload_custom_libs():
    import loss_functions
    import models
    import dataprocessor
    importlib.reload(loss_functions)
    importlib.reload(models)
    importlib.reload(dataprocessor)


# 加载文件
macd_timescales = [(8, 24), (16, 28), (32, 96)]
rtn_timescales = [1, 21, 63, 126, 252]
timesteps = 126

folder_path = "data"
files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
files = [
    file
    for file in files
    if file
    not in (
        "CC00.NYB.xlsx",
        "LB00.CME.xlsx",
        "ES00.CME.xlsx",
        "NQ00.CME.xlsx",
        "YM00.CBT.xlsx",
        "SP00.CME.xlsx",
    )
]
data_list = []


data_list = process_data_list(files, macd_timescales, rtn_timescales, test=False)
# 试试全部的数据。。

tmp = []
for data in tqdm(data_list):
    price_series = data["close"]
    target = price_series.to_numpy().reshape((-1, 1))
    segment_list = get_segment_points(target, l_max=63, mu=0.999)
    segment_list = [data.iloc[start : end, :] for start, end in segment_list]
    tmp.extend(segment_list)