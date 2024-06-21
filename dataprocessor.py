import numpy as np
import pandas as pd 
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

MACD_timescales = [(8, 24), (16, 28), (32, 96)]
RTN_timescales = [1, 21, 63, 126, 252]


def process_data_list(files: list[str], macd_timescales: list=MACD_timescales, rtn_timescales: list=RTN_timescales, test: bool = False) -> list[pd.DataFrame]:
    """
    处理文件列表并生成包含特征的数据帧列表。

    Args:
        files (list[str]): 文件路径列表。

    Returns:
        list[pd.DataFrame]: 包含处理后数据的数据帧列表。
    """
    data_list = []
    for file in tqdm(files, desc="处理文件中。。"):
        if file.endswith("xlsx"):
            data = pd.read_excel(f"data/{file}")[["日期", "收盘价(元)"]]
            side_info = file.replace(".xlsx", "")
        elif file.endswith("parquet"):
            data = pd.read_parquet(f"data/{file}")[["日期", "收盘价(元)"]]
            side_info = file.replace(".parquet", "")
        elif file.endswith("csv"):
            data = pd.read_csv(f"data/{file}")[["日期", "收盘价(元)"]]
            side_info = file.replace(".csv", "")

        data = data.rename(columns={"日期": "date", "收盘价(元)": "close"}).sort_values("date")
        data["side_info"] = side_info
        try:
            data = generate_features(data, macd_timescales, rtn_timescales)
        except:
            print(f"{side_info} 中含有空值，其中的数据没有录入列表")
            continue
        if test: data = data.iloc[:64, :]
        data_list.append(data)
    return data_list


def generate_features(data: pd.DataFrame, macd_timscales: list, rtn_timscales: list) -> pd.DataFrame:
    """
    计算MACD指标。

    Args:
        data (pd.DataFrame): 包含日期和收盘价的数据。
        t (int): 时间窗口, 用于计算滚动标准差。
        s (int): 短期指数平滑移动平均线(EMA)的平滑参数。
        l (int): 长期指数平滑移动平均线(EMA)的平滑参数。

    Returns:
        pd.DataFrame: 包含日期和MACD值的数据。
    """
    data = data.sort_values("date")
    data["rtn_next_day"] = data["close"].shift(-1) / data["close"] - 1
    data["rtn"] = data["close"] / data["close"].shift(1) - 1
    
    close_series = data["close"]
    date_series = data["date"]
    rtn_series = data["rtn"]
    data = data.set_index("date")
    
    if close_series.isnull().any() or close_series.isna().any():
        raise ValueError("收盘数据中存在 null 或者 na, 检查输入的数据")
    
    exp = lambda series, alpha: series.ewm(alpha=alpha)
    for s, l in macd_timscales:
        m = []
        for i in range(60, len(data)):
            tmp_close_series = close_series[i - 60:i]
            std = tmp_close_series.std()
            ewma_s = exp(tmp_close_series, 1 / s).mean().iloc[-1]
            ewma_l = exp(tmp_close_series, 1 / l).mean().iloc[-1]
            if std == 0: 
                raise ValueError("Problem")
                
            m_t = (ewma_s - ewma_l) / std
            m.append(m_t)
        m = pd.Series(m, index=date_series[60:])
        m_std = m.rolling(252).std()
        data[f"macd_{s}_{l}"] = pd.Series(m / m_std, index=date_series)
            
    sigma = []
    for i in range(60, len(data)):
        tmp_rtn_series = rtn_series[i - 60:i]
        ewmstd = exp(tmp_rtn_series, 0.2057).std().iloc[-1]
        sigma.append(ewmstd)
    sigma = pd.Series(sigma, index=date_series[60:])
    data["sigma"] = pd.Series(sigma)
    
    for i in rtn_timscales:
        rtn = close_series / close_series.shift(i) - 1
        rtn = pd.Series(rtn.values, index=date_series.values)
        normalized_rtn = rtn / data["sigma"] / i ** 0.5
        data[f"rtn_{i}"] = pd.Series(normalized_rtn, index=date_series)
        
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    return data


def generate_tensors(data_list: list[pd.DataFrame], timesteps: int, contain_next_day_rtn: bool = False, return_labels: bool = False):
    """
    生成张量序列。

    Args:
        data_list (list[pd.DataFrame]): 数据帧列表。
        timesteps (int): 时间步长。
        contain_next_day_rtn (bool): 是否包含“rtn_next_day”列。默认值为False。
        return_labels (bool): 是否返回标签数据。默认值为True。

    Returns:
        tuple: 包含特征序列、日期和附加信息的张量。如果 return_label_df 为 True, 还返回 rtn_next_day 和 sigma。
    """
    feature_sequences = []
    dates = []
    side_info = []
    rtn_next_day = []
    std = []
    for data in data_list:
        data["date"] = pd.to_datetime(data["date"]).dt.strftime('%Y-%m-%d')
        feature_cols = data.columns.to_list()
        for col in ("date", "sigma", "side_info", "rtn_next_day", "close", "rtn"): 
            if contain_next_day_rtn and col == "rtn_next_day": continue
            feature_cols.remove(col)

        def create_sequences(data, timesteps):
            sequences = []
            for i in range(len(data) - timesteps + 1):
                sequence = data.iloc[i:i + timesteps]
                sequences.append(sequence)
            return sequences

        sequences = create_sequences(data, timesteps)
        feature_sequences = feature_sequences + [seq[feature_cols].values for seq in sequences]
        dates = dates + [seq['date'].values[-1] for seq in sequences]
        side_info = side_info + [seq['side_info'].values[-1] for seq in sequences]
        rtn_next_day = rtn_next_day + [seq['rtn_next_day'].values[-1] for seq in sequences]
        std = std + [seq['sigma'].values[-1] for seq in sequences]
        
    feature_sequences = np.array(feature_sequences)
    dates = np.array(dates)
    side_info = np.array(side_info)
    rtn_next_day = np.array(rtn_next_day)
    std = np.array(std)
    
    feature_sequences_tensor = tf.convert_to_tensor(feature_sequences, dtype=tf.float32)
    dates_tensor = tf.convert_to_tensor(dates, dtype=tf.string)
    side_info_tensor = tf.convert_to_tensor(side_info, dtype=tf.string)
    if not return_labels:
        return (feature_sequences_tensor, dates_tensor, side_info_tensor)
    return (feature_sequences_tensor, dates_tensor, side_info_tensor), (rtn_next_day, std)


def side_info_tensor_encoder(string_tensor: tf.Tensor, return_dict = True):
    """
    对字符串张量进行one-hot编码, 并返回one-hot编码后的张量及对应的映射表。

    Args:
        string_tensor (tf.Tensor): 输入的字符串张量。

    Returns:
        tuple: one-hot编码后的张量和对应的映射表。
    """
    string_list = [s.decode('utf-8') for s in string_tensor.numpy()]

    # 初始化Tokenizer
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(string_list)

    # 将字符串转换为整数索引
    integer_encoded = tokenizer.texts_to_sequences(string_list)
    integer_encoded = [item[0] for item in integer_encoded]
    one_hot_encoded = to_categorical(integer_encoded)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    one_hot_encoded_tensor = tf.convert_to_tensor(one_hot_encoded, dtype=tf.float32)
    if return_dict:
        return one_hot_encoded_tensor, index_word
    return word_index