import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

MACD_timescales = [(8, 24), (16, 28), (32, 96)]
RTN_timescales = [1, 21, 63, 126, 252]


def process_data_list(
    files: list[str],
    macd_timescales: list = MACD_timescales,
    rtn_timescales: list = RTN_timescales,
    test: bool = False,
) -> list[pd.DataFrame]:
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

        data = data.rename(columns={"日期": "date", "收盘价(元)": "close"}).sort_values(
            "date"
        )
        data["side_info"] = side_info
        try:
            data = generate_features(data, macd_timescales, rtn_timescales)
        except:
            print(f"{side_info} 中含有空值，其中的数据没有录入列表")
            continue
        if test:
            data = data.iloc[:64, :]
        data_list.append(data)
    return data_list


def generate_features(
    data: pd.DataFrame, macd_timscales: list, rtn_timscales: list
) -> pd.DataFrame:
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
            tmp_close_series = close_series[i - 60 : i]
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
        tmp_rtn_series = rtn_series[i - 60 : i]
        ewmstd = exp(tmp_rtn_series, 0.2057).std().iloc[-1]
        sigma.append(ewmstd)
    sigma = pd.Series(sigma, index=date_series[60:])
    data["sigma"] = pd.Series(sigma)

    for i in rtn_timscales:
        rtn = close_series / close_series.shift(i) - 1
        rtn = pd.Series(rtn.values, index=date_series.values)
        normalized_rtn = rtn / data["sigma"] / i**0.5
        data[f"rtn_{i}"] = pd.Series(normalized_rtn, index=date_series)

    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    return data


def generate_tensors(
    data_list: list[pd.DataFrame],
    time_steps: int,
    encoder_type: str,
    contain_next_day_rtn: bool = False,
    return_map: bool = False,
):
    """
    生成特征张量、日期张量和辅助信息张量。

    Args:
        data_list (list[pd.DataFrame]): 包含数据的DataFrame列表。
        time_steps (int): 时间步长。
        encoder_type (str): 编码器类型。
        contain_next_day_rtn (bool): 是否包含次日回报率。
        return_map (bool): 是否返回编码映射。

    Returns:
        tuple: 包含特征张量、日期张量、辅助信息张量、次日回报率和标准差的元组。
    """
    feature_sequences = []
    dates = []
    side_info = []
    rtn_next_day = []
    std = []
    for data in data_list:
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        feature_cols = data.columns.to_list()
        for col in ("date", "sigma", "side_info", "rtn_next_day", "close", "rtn"):
            if contain_next_day_rtn and col == "rtn_next_day":
                continue
            feature_cols.remove(col)

        def create_sequences(data, time_steps):
            """
            创建时间序列数据。

            Args:
                data (pd.DataFrame): 输入数据。
                time_steps (int): 时间步长。

            Returns:
                list: 序列数据列表。
            """
            sequences = []
            for i in range(len(data) - time_steps + 1):
                sequence = data.iloc[i : i + time_steps]
                sequences.append(sequence)
            return sequences

        sequences = create_sequences(data, time_steps)
        feature_sequences = feature_sequences + [
            seq[feature_cols].values for seq in sequences
        ]
        dates = dates + [seq["date"].values[-1] for seq in sequences]
        side_info = side_info + [seq["side_info"].values[-1] for seq in sequences]
        rtn_next_day = rtn_next_day + [
            seq["rtn_next_day"].values[-1] for seq in sequences
        ]
        std = std + [seq["sigma"].values[-1] for seq in sequences]

    feature_sequences = np.array(feature_sequences)
    dates = np.array(dates)
    side_info = np.array(side_info)
    rtn_next_day = np.array(rtn_next_day)
    std = np.array(std)

    feature_sequences_tensor = tf.convert_to_tensor(feature_sequences, dtype=tf.float32)
    dates_tensor = tf.convert_to_tensor(dates, dtype=tf.string)
    side_info_tensor = tf.convert_to_tensor(side_info, dtype=tf.string)

    if encoder_type == "one-hot":
        print("one-hot 编码中...")
        side_info_tensor, side_info_map = side_info_one_hot_encoder(
            side_info_tensor, time_steps=time_steps
        )
    else:
        raise NotImplementedError("除 one-hot 之外的方法暂未实现")

    if return_map:
        return (
            (feature_sequences_tensor, dates_tensor, side_info_tensor),
            (rtn_next_day, std),
            side_info_map,
        )
    return (
        (feature_sequences_tensor, dates_tensor, side_info_tensor),
        (rtn_next_day, std)
    )


def side_info_one_hot_encoder(
    string_tensor: tf.Tensor, time_steps: int, return_dict=True
):
    """
    对字符串张量进行one-hot编码, 并返回one-hot编码后的三维张量及对应的映射表。

    Args:
        string_tensor (tf.Tensor): 输入的字符串张量。

    Returns:
        tuple: one-hot编码, 并扩展后的三维张量和对应的映射表。
    """
    string_list = [s.decode("utf-8") for s in string_tensor.numpy()]

    # 初始化Tokenizer
    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(string_list)

    # 将字符串转换为整数索引
    integer_encoded = tokenizer.texts_to_sequences(string_list)
    integer_encoded = [item[0] for item in integer_encoded]
    one_hot_encoded = to_categorical(integer_encoded)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    one_hot_encoded_tensor = tf.convert_to_tensor(one_hot_encoded, dtype=tf.float32)
    expanded_info = tf.expand_dims(
        one_hot_encoded_tensor, axis=1
    )  # Shape will be [batch, 1, feature]
    tiled_tensor = tf.tile(expanded_info, [1, time_steps, 1])
    if return_dict:
        return tiled_tensor, index_word
    return tiled_tensor


def data_binder(
    context_set: tuple[tf.Tensor],
    target_set: tuple[tf.Tensor],
    labels: tuple[np.array],
    return_dates: bool = False,
):
    """
    绑定上下文和目标数据集，并匹配相应的标签和日期。

    Args:
        context_set (tuple[tf.Tensor]): 包含上下文特征、上下文日期和辅助信息的元组。
        target_set (tuple[tf.Tensor]): 包含目标特征、目标日期和目标信息的元组。
        labels (tuple[np.array]): 包含均值和标准差的标签元组。
        return_dates (bool): 是否返回日期信息。

    Returns:
        tuple: 包含上下文特征、目标特征、标签和（可选）日期的元组。
    """
    target_features, target_dates, target_info = target_set
    context_features, context_dates, side_info = context_set
    mean_set, std_set = labels

    def date_to_int(date):
        date_int = tf.strings.to_number(
            tf.strings.regex_replace(date, "-", ""), tf.int32
        )
        return date_int

    def find_matching_context(
        target_date, target_info, context_dates, context_features, side_info
    ):
        """
        查找与目标日期和信息匹配的上下文特征和日期。

        Args:
            target_date (tf.Tensor): 目标日期。
            target_info (tf.Tensor): 目标信息。
            context_dates (tf.Tensor): 上下文日期。
            context_features (tf.Tensor): 上下文特征。
            side_info (tf.Tensor): 辅助信息。

        Returns:
            tuple: 匹配的上下文特征、上下文日期和辅助信息。
        """
        mask_date = tf.math.greater(
            date_to_int(target_date), date_to_int(context_dates)
        )
        mask_info = tf.reduce_all(
            tf.math.equal(target_info[0], side_info[:, -1, :]), axis=1
        )
        mask_info = tf.logical_not(mask_info)
        mask = tf.logical_and(mask_date, mask_info)
        valid_indices = tf.where(mask)
        if tf.size(valid_indices) > 0:
            random_index = tf.random.uniform(
                shape=[], minval=0, maxval=tf.size(valid_indices), dtype=tf.int32
            )
            selected_index = valid_indices[random_index, 0]
            return (
                context_features[selected_index],
                context_dates[selected_index],
                side_info[selected_index],
            )
        else:
            return None, None, None

    matched_context_features = []
    matched_context_dates = []
    matched_side_info = []
    matched_target_features = []
    matched_target_dates = []
    matched_target_info = []
    target_distribution = []

    for i in tqdm(range(len(target_dates)), desc="处理日期中.."):
        target_date = target_dates[i]
        target_inf = target_info[i]
        target_feature = target_features[i]
        mean, std = mean_set[i], std_set[i]

        context_feature, context_date, side_inf = find_matching_context(
            target_date, target_inf, context_dates, context_features, side_info
        )

        if context_feature is not None:
            matched_context_features.append(context_feature)
            matched_context_dates.append(context_date)
            matched_side_info.append(side_inf)
            matched_target_features.append(target_feature)
            matched_target_dates.append(target_date)
            matched_target_info.append(target_inf)
            target_distribution.append((mean, std))

    if not matched_context_features:
        matched_context_features = tf.constant([])
        matched_context_dates = tf.constant([])
        matched_side_info = tf.constant([])
        raise ValueError("数据集中没有课配对的数据")

    x_c_rtn = tf.stack(matched_context_features)
    x_c = x_c_rtn[:, :, 1:]
    s_c = tf.stack(matched_side_info)
    x = tf.stack(matched_target_features)
    s = tf.stack(matched_side_info)
    mean_std = tf.stack(target_distribution)
    x_c_date = tf.stack(matched_context_dates)
    x_dates = tf.stack(matched_context_dates)
    if return_dates:
        return (x_c, x_c_rtn, s_c), (x, s), mean_std, (x_c_date, x_dates)
    return (x_c, x_c_rtn, s_c), (x, s), mean_std
