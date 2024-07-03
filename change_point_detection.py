import numpy as np
import pandas as pd
import gpflow
from tqdm import tqdm

lbw = 21
l_max, mu = 21, 0.9
l_max, mu = 63, 0.99
l_min = 5
steepness = 1


def get_segment_points(
    Y: np.ndarray,
    l_max: int,
    mu: float,
    l_min: int = 5,
    lbw: int = 21,
    steepness: int = 1,
) -> list[tuple]:
    """
    使用高斯过程和变点检测算法在时间序列数据中寻找分段点。

    参数：
    Y (np.ndarray): 输入的时间序列数据。
    l_max (int): 允许的最大段长度。
    mu (float): 用于判断变点的阈值。
    l_min (int, 可选): 允许的最小段长度, 默认值为5。
    lbw (int, 可选): 滑动窗口的长度, 默认值为21。
    steepness (int, 可选): 变点核函数的陡峭度, 默认值为1。

    返回：
    list[tuple]: 找到的分段点列表, 每个分段点表示为 (start, end) 的元组。
    """

    loc, v = None, 0
    t, t_1 = len(Y), len(Y)
    X = np.arange(len(Y)).reshape(-1, 1).astype(float)
    R = []
    while t >= 0:
        if t - lbw < 0:
            break

        x = X[t - lbw : t]
        y = Y[t - lbw : t]

        location = x[int((lbw) // 2)][0]

        kernel = gpflow.kernels.Matern32()
        k1 = gpflow.kernels.Matern32()
        k2 = gpflow.kernels.Matern32()

        # 初始化，拟合一个高斯变点模型
        changepoint_kernel = gpflow.kernels.ChangePoints(
            [k1, k2], locations=[location], steepness=steepness
        )
        gpr_model_base = gpflow.models.GPR(
            data=(x, y), kernel=changepoint_kernel, mean_function=None
        )
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            gpr_model_base.training_loss, variables=gpr_model_base.trainable_variables
        )
        nlml_C = gpr_model_base.training_loss().numpy()

        # 初始化，拟合一个高斯过程模型
        kernel = gpflow.kernels.Matern32()
        model = gpflow.models.GPR(data=(x, y), kernel=kernel, mean_function=None)
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(
            model.training_loss, model.trainable_variables, options=dict(maxiter=100)
        )
        nlml_M = model.training_loss().numpy()

        prev_loc = loc
        loc = changepoint_kernel.locations.numpy()[0]
        if loc < x[-1][0] and loc > x[0][0]:
            # 成功拟合
            L_C = np.exp(-nlml_C)
            L_M = np.exp(-nlml_M)
            v = L_C / (L_C + L_M)
        else:
            if prev_loc is not None:
                loc = prev_loc - 1
            else:
                loc = x[-1][0] - 1  # 如果是第一次拟合失败, 使用当前窗口的最后一个时间点

        if v > mu:
            t_0 = int(np.ceil(loc))
            if t_1 - t_0 >= l_min:
                R.append((t_0, t_1))
                t_1 = t_0 - 1
            t = int(np.floor(loc)) - 1
        else:
            t = t - 1
            if t_1 - t > l_max:
                t = t_1 - l_max
            if t_1 - t == l_max:
                R.append((t, t_1))
                t_1 = t
        if t % 100 == 0:
            print(t)
    R.append((0, R[-1][0] - 1))
    R = list(reversed(R))
    return R


def get_segment_list(data_list: list[pd.DataFrame]):
    gaussion_process_list = []
    for data in tqdm(data_list):
        price_series = data["close"]
        target = price_series.to_numpy().reshape((-1, 1))
        segment_list = get_segment_points(target, l_max=63, mu=0.999)
        segment_list = [data.iloc[start : end, :] for start, end in segment_list]
        gaussion_process_list.extend(segment_list)
    
    return gaussion_process_list