import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, ELU, Add, Softmax, LayerNormalization
from tensorflow.keras.models import Model
import tensorflow.keras as keras


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
    return data


def generate_tf_dataset(data_list: list[pd.DataFrame], timesteps: int, context: bool = False) -> tf.data.Dataset:
    """
    生成TensorFlow数据集。

    Args:
        data_list (list[pd.DataFrame]): 数据列表, 每个元素为一个数据帧。
        timesteps (int): 时间步数, 用于创建序列。
        context (bool): 是否包含次日回报率信息。如果为True, 保留 'rtn_next_day' 列。

    Returns:
        tf.data.Dataset: 生成的TensorFlow数据集, 包含特征序列、日期和辅助信息。
    """
    feature_sequences = []
    dates = []
    side_info = []
    for data in data_list:
        if not context:
            X = data.dropna(axis=0).drop(columns=["rtn_next_day", "close", "rtn", "sigma"])
        else:
            X = data.dropna(axis=0).drop(columns=["close", "rtn", "sigma"])
        X["date"] = pd.to_datetime(X["date"]).dt.strftime('%Y-%m-%d')
        feature_cols = X.columns.to_list()
        for col in ("date", "side_info"): feature_cols.remove(col)

        def create_sequences(data, timesteps):
            sequences = []
            for i in range(len(data) - timesteps + 1):
                sequence = data.iloc[i:i + timesteps]
                sequences.append(sequence)
            return sequences

        sequences = create_sequences(X, timesteps)

        feature_sequences = feature_sequences + [seq[feature_cols].values for seq in sequences]
        dates = dates + [seq['date'].values[-1] for seq in sequences]
        side_info = side_info + [seq['side_info'].values[-1] for seq in sequences]
        
    feature_sequences = np.array(feature_sequences)
    dates = np.array(dates)
    side_info = np.array(side_info)
    dataset = tf.data.Dataset.from_tensor_slices((feature_sequences, {"date": dates, "side_info": side_info}))
    return dataset


def encode_context_info(dataset: tf.data.Dataset) -> tf.Tensor:
    """
    对上下文信息进行编码,并返回一个包含编码后的上下文信息的Tensor。

    Args:
        dataset (tf.data.Dataset): TensorFlow数据集,其中包含特征和上下文信息。

    Returns:
        tf.Tensor: 编码后的上下文信息Tensor,形状与输入数据集的特征部分匹配。
    """
    
    context_info_values = []
    encoded_data = []

    print("编码中...")
    for features, context_info in dataset:
        side_info = context_info['side_info'].numpy().decode('utf-8')
        context_info_values.append(side_info)

    unique_context_info = list(set(context_info_values))
    context_info_map = {val: idx for idx, val in enumerate(unique_context_info)}
    num_classes = len(unique_context_info)

    def one_hot_encode(value):
        one_hot = np.zeros(num_classes)
        one_hot[context_info_map[value]] = 1
        return one_hot

    for side_info in context_info_values:
        one_hot_encoded = one_hot_encode(side_info)
        encoded_data.append(one_hot_encoded)

    encoded_data = tf.ragged.constant(encoded_data).to_tensor(default_value=0.0)

    features, _ = next(iter(dataset))
    shape = features.shape
    expanded_tensor = tf.expand_dims(encoded_data, axis=1)
    encoded_data = tf.tile(expanded_tensor, [1, shape[0], 1])
    return encoded_data