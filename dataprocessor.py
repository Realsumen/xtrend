import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from change_point_detection import *

MACD_timescales = [(8, 24), (16, 28), (32, 96)]
RTN_timescales = [1, 21, 63, 126, 252]


def process_file(args):
    file, macd_timescales, rtn_timescales, test = args
    
    # 判断文件的类型，进行处理
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
        print(f"{side_info} 中含有空值, 其中的数据没有录入列表")
        return None
    if test is not None:
        # 如果 test 不为空的话，分段生成变点数据帧
        if len(data) < test[0]:
            data =  None
        elif len(data) > test[0] and len(data) < test[1]:
            data = data.iloc[test[0]:, :]
        else:
            data = data.iloc[test[0]:test[1], :]
    return data


def process_data_list(
    files: list[str],
    macd_timescales: list = MACD_timescales,
    rtn_timescales: list = RTN_timescales,
    test: int = None,
) -> list[pd.DataFrame]:
    """
    处理文件列表并生成包含特征的数据帧列表。

    args:
        files (list[str]): 文件路径的列表。
        macd_timescales (list, optional): MACD的时间尺度列表。默认为MACD_timescales。
        rtn_timescales (list, optional): 回报率的时间尺度列表。默认为RTN_timescales。
        test (int, optional): 测试参数/分段进行变点检测, 默认为None。

    return:
        list[pd.DataFrame]: 包含处理后数据的数据帧列表, 每个数据帧是一个资产的一个无变点片段
    """
    data_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_file, (file, macd_timescales, rtn_timescales, test)) for file in files]
        for future in futures:
            result = future.result()
            if result is not None:
                data_list.append(result)
    
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
                raise ValueError("多日价格相同导致标准差为0, 检查输入的收盘数据")

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
        normalized_rtn = rtn / data["sigma"] / i ** 0.5
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
    for data in tqdm(data_list, desc="生成张量, 并对类别信息进行one-hot 编码"):
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
        rtn_next_day = rtn_next_day + [seq["rtn_next_day"].values for seq in sequences]
        std = std + [seq["sigma"].values for seq in sequences]

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
        (rtn_next_day, std),
    )


def generate_context_tensors(data_list: list[pd.DataFrame], method: str, **params):
    if method == "Time":
        result = generate_tensors(data_list, **params)
        return result
    if method == "Gaussian":
        # 生成 context set
        gaussion_process_list = params["gaussion_process_list"]
        map = params["map"]
        
        feature_cols = gaussion_process_list[0].columns.to_list()
        for col in ("date", "sigma", "side_info", "close", "rtn"):
            feature_cols.remove(col)

        features = [
            df[feature_cols].values for df in gaussion_process_list if len(df) > 0
        ]
        date_list = [
            str(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").values[-1])
            for df in gaussion_process_list
            if len(df) > 0
        ]
        side_info = [
            df["side_info"].iloc[0] for df in gaussion_process_list if len(df) > 0
        ]
        rtn_next_day = [
            df["rtn_next_day"] for df in gaussion_process_list if len(df) > 0
        ]
        std = [df["sigma"] for df in gaussion_process_list if len(df) > 0]

        feature_sequences_tensor = tf.ragged.constant(features, dtype=tf.float32)
        dates_tensor = tf.constant(date_list, dtype=tf.string)
        side_info_tensor = tf.constant(side_info, dtype=tf.string)
        rtn_next_day = tf.ragged.constant(rtn_next_day)
        std = tf.ragged.constant(std)

        def pad_ragged_tensors(tensor):
            padded = tensor.to_tensor(0.0)
            return tf.reverse(padded, [-1])

        feature_sequences_tensor = pad_ragged_tensors(feature_sequences_tensor)
        rtn_next_day = pad_ragged_tensors(rtn_next_day)
        std = pad_ragged_tensors(std)

        side_info_tensor, _ = side_info_one_hot_encoder(
            side_info_tensor,
            time_steps=feature_sequences_tensor.shape[1],
            word_index=map,
            return_dict=True,
        )
        context_set = (feature_sequences_tensor, dates_tensor, side_info_tensor)
        return context_set


def side_info_one_hot_encoder(
    
    string_tensor: tf.Tensor,
    word_index: dict = None,
    return_dict=True,
    time_steps: int = None,
    shape_tensor=None,
):
    """
    对字符串张量进行one-hot编码, 并返回one-hot编码后的三维张量及对应的映射表。

    Args:
        string_tensor (tf.Tensor): 输入的字符串张量。

    Returns:
        tuple: one-hot编码, 并扩展后的三维张量和对应的映射表。
    """
    string_list = [s.decode("utf-8") for s in string_tensor.numpy()]
    if word_index is None:
        tokenizer = Tokenizer(filters="", lower=False)
        tokenizer.fit_on_texts(string_list)
        word_index = tokenizer.word_index

    # 将字符串转换为整数索引
    integer_encoded = [word_index.get(word, 0) for word in string_list]
    one_hot_encoded = to_categorical(integer_encoded, num_classes=len(word_index) + 1)
    one_hot_encoded_tensor = tf.convert_to_tensor(one_hot_encoded, dtype=tf.float32)
    expanded_info = tf.expand_dims(one_hot_encoded_tensor, axis=1)
    if time_steps != None:
        if shape_tensor != None:
            raise ValueError("同时传入了timesteps和shape_tensor!")
        expanded_info = tf.expand_dims(one_hot_encoded_tensor, axis=1)
        tiled_tensor = tf.tile(expanded_info, [1, time_steps, 1])
    elif shape_tensor != None:
        expanded_info = tf.repeat(one_hot_encoded_tensor, repeats=shape_tensor, axis=0)
        tiled_tensor = tf.RaggedTensor.from_row_lengths(expanded_info, shape_tensor)
    else:
        raise ValueError(
            "未指定时间步维度, 对于规则张量请传入时间步timespteps; 对并于不规则张量, 请传入shape_tensor"
        )
    if return_dict:
        return tiled_tensor, word_index
    return tiled_tensor


def date_to_int(date):
    """
    Args:
        date (tf.Tensor): 日期字符串张量。
    Returns:
        tf.Tensor: 转换后的整数日期张量。
    """
    return tf.strings.to_number(tf.strings.regex_replace(date, "-", ""), tf.int64)


def gaussian_data_binder(
    data_list: list[pd.DataFrame],
    target_set: tuple[tf.Tensor],
    labels: tuple[np.array],
    map: int,
    asset_num: int,
    context_num: int,
    gaussion_process_list: list = None,
):
    if gaussion_process_list is None:
        gaussion_process_list = get_segment_list(data_list=data_list)

    # 生成 context set
    context_set = generate_context_tensors(
        data_list=data_list,
        method="Gaussian",
        gaussion_process_list=gaussion_process_list,
        map = map
    )

    # 把 target 绑定按照asset_num资产数目绑定
    features, dates, context = target_set
    rtn, std = labels

    unique_dates, _, counts = tf.unique_with_counts(dates)
    twice_dates = tf.boolean_mask(unique_dates, counts == asset_num)

    # Create a mapping from dates to indices
    date_to_indices = {
        date.numpy(): tf.where(dates == date).numpy().flatten() for date in twice_dates
    }

    # Gather features and contexts for these dates, ensuring the order is maintained
    ordered_features, ordered_side_info, ordered_rtn_std = [], [], []

    for date in twice_dates:
        indices = date_to_indices[date.numpy()]
        ordered_features.append(tf.gather(features, indices))
        ordered_side_info.append(tf.gather(context, indices))
        ordered_rtn_std.append((tf.gather(rtn, indices), tf.gather(std, indices)))

    ordered_features = tf.stack(ordered_features)
    ordered_side_info = tf.stack(ordered_side_info)
    target_set = (ordered_features, twice_dates, ordered_side_info)
    ordered_rtn_std = tf.transpose(tf.stack(ordered_rtn_std), perm=[0, 2, 3, 1])

    # 把 context 按照 context_num 数目绑定

    feature_sequences_tensor = context_set[0]
    dates_tensor = context_set[1]
    side_info_tensor = context_set[2]
    dates_tensor_int = [date_to_int(date) for date in dates_tensor.numpy()]

    new_feature_sequences = []
    new_side_info_sequences = []
    binded_dates = []
    for date in twice_dates.numpy():
        # 找到比当前日期小的日期
        smaller_dates_indices = tf.where(dates_tensor_int < date_to_int(date))
        if len(smaller_dates_indices) > context_num:
            # 选择一个随机的日期
            chosen_index = tf.random.shuffle(smaller_dates_indices)
            chosen_index = tf.reshape(chosen_index, [-1])
            # 提取对应的 feature 和 side_info
            new_feature_sequences.append(
                tf.stack(
                    [
                        feature_sequences_tensor[chosen_index[i]]
                        for i in range(context_num)
                    ]
                )
            )
            new_side_info_sequences.append(
                tf.stack(
                    [side_info_tensor[chosen_index[i]] for i in range(context_num)]
                )
            )
            binded_dates.append(date)

    binded_feature_sequences_tensor = tf.stack(new_feature_sequences)
    binded_side_info_sequences_tensor = tf.stack(new_side_info_sequences)
    binded_dates_tensor = tf.constant(binded_dates)

    context_set = (
        binded_feature_sequences_tensor,
        binded_dates_tensor,
        binded_side_info_sequences_tensor,
    )

    # context 和 target 配对
    context_dates_set = set(context_set[1].numpy())

    # 筛选 target_dates 中属于 context_dates 的索引
    indices = [
        i for i, date in enumerate(target_set[1].numpy()) if date in context_dates_set
    ]

    # 根据已绑定的 target_set 日期索引筛选 target_set， 没绑定的直接舍弃
    filtered_target_feature_sequences = tf.gather(target_set[0], indices)
    filtered_target_dates = tf.gather(target_set[1], indices)
    filtered_target_side_info = tf.gather(target_set[2], indices)
    labels = tf.gather(ordered_rtn_std, indices)

    target_set = (
        filtered_target_feature_sequences,
        filtered_target_dates,
        filtered_target_side_info,
    )

    return target_set, context_set, labels




"""
    下面的方法data_binder_do_not_use 用来生成Time-equivalent hidden state context set, 暂时弃用
"""



def data_binder_do_not_use(
    context_set: tuple[tf.Tensor],
    target_set: tuple[tf.Tensor],
    labels: tuple[np.array],
    batch_size: int,
):
    """
    将上下文数据和目标数据绑定在一起, 并生成一个TensorFlow数据集。

    Args:
        context_set (tuple[tf.Tensor]): 上下文数据集, 包括特征、日期和辅助信息。
        target_set (tuple[tf.Tensor]): 目标数据集, 包括特征、日期和信息。
        labels (tuple[np.array]): 目标数据的标签, 包括收益率和标准差。
        batch_size (int): 批处理大小。

    Returns:
        tf.data.Dataset: 生成的TensorFlow数据集。
    """
    def key_func(x_c, x_c_rtn, s_c, x, s, rtn_std, x_c_date, x_dates):
        """
        databinder() 中，获取用于分组的键 (相同日期一组)。
        """
        return date_to_int(x_dates)


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

        mask_date = tf.math.greater(date_to_int(target_date), date_to_int(context_dates))
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
    target_features, target_dates, target_info = target_set
    context_features, context_dates, side_info = context_set
    rtn_set, std_set = labels

    matched_context_features = []
    matched_context_dates = []
    matched_side_info = []
    matched_target_features = []
    matched_target_dates = []
    matched_target_info = []
    target_labels = []

    for i in tqdm(range(len(target_dates)), desc="处理日期中.."):
        target_date = target_dates[i]
        target_inf = target_info[i]
        target_feature = target_features[i]
        rtn, std = rtn_set[i], std_set[i]

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
            target_labels.append((rtn, std))

    if not matched_context_features:
        matched_context_features = tf.constant([])
        matched_context_dates = tf.constant([])
        matched_side_info = tf.constant([])
        raise ValueError("数据集中没有可配对的数据")

    x_c_rtn = tf.stack(matched_context_features)
    x_c = x_c_rtn[:, :, 1:]
    s_c = tf.stack(matched_side_info)
    x = tf.stack(matched_target_features)
    s = tf.stack(matched_target_info)
    rtn_std = tf.stack(target_labels)
    x_c_date = tf.stack(matched_context_dates)
    x_dates = tf.stack(matched_target_dates)
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_c, x_c_rtn, s_c, x, s, rtn_std, x_c_date, x_dates)
    )

    def filter_batch(dataset, batch_size):
        """
        过滤掉批处理大小不符合要求的数据。

        Args:
            dataset (tf.data.Dataset): 原始数据集。
            batch_size (int): 批处理大小。

        Returns:
            tf.data.Dataset: 过滤后的数据集。
        """
        batched_dataset = dataset.batch(batch_size)
        return batched_dataset.filter(
            lambda x_c, x_c_rtn, s_c, x, s, rtn_std, x_c_date, x_dates: tf.shape(x_c)[0]
            == batch_size
        )

    def reduce_func(key, dataset):
        """
        进行分组后的归约操作。
        """
        return filter_batch(dataset, batch_size)

    dataset = dataset.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size
    )

    return dataset

