import numpy as np
import tensorflow as tf

def sharpe_loss(
    positions: tf.Tensor, rtn_std: tf.Tensor, target_std: float, warm_up: int
):  
    """
    计算夏普比率损失函数。

    Args:
        positions (tf.Tensor): 资产仓位张量。
        rtn_std (tf.Tensor): 资产的收益率和标准差张量。
        target_std (float): 目标标准差。
        warm_up (int): 预热阶段的时间步数。

    Returns:
        tf.Tensor: 夏普比率损失值。
    """
    target_std = tf.cast(target_std, tf.float64)
    practice_timesteps = positions.shape[2] - warm_up
    assets_daily_rtn = rtn_std[:, :, :, 0] / rtn_std[:, :, :, 1] * target_std * positions
    portfolio_daily_rtn = tf.reduce_sum(assets_daily_rtn, axis=1)
    portfolio_daily_rtn = portfolio_daily_rtn[:, -practice_timesteps:]
    mean = tf.math.reduce_mean(portfolio_daily_rtn, axis = 1)
    std = tf.math.reduce_variance(portfolio_daily_rtn, axis = 1) ** 0.5
    loss = -(252**0.5) * mean / std
    return loss


def mle_loss(
    properties: tf.Tensor, rtn_std: tf.Tensor, target_std: float, warm_up: int
):
    """
    计算最大似然估计 (MLE) 损失函数。

    Args:
        properties (tf.Tensor): 资产次日收益率分布特征张量。
        rtn_std (tf.Tensor): 资产的收益率和标准差张量。
        target_std (float): 目标标准差。
        warm_up (int): 暖启动阶段的时间步数。

    Returns:
        tf.Tensor: MLE损失值。
    """
    _, asset_num, time_steps, _  = properties.shape
    practice_timesteps = time_steps - warm_up
    mean, std = rtn_std[:, :, -practice_timesteps:, 0], rtn_std[:, :, -practice_timesteps:, 1]
    assets_daily_change_pct = rtn_std[:, :, -practice_timesteps:, 0] / rtn_std[:, :, -practice_timesteps:, 1] * target_std
    log_likelihood = tf.reduce_sum(
        -tf.math.log((2.0 * tf.constant(np.pi, tf.float64)) ** 0.5 * std)
        - ((assets_daily_change_pct - mean) ** 2 / (2 * std ** 2)), axis=[1, 2]
    )
    loss = log_likelihood * -1 / asset_num / practice_timesteps
    return loss

def joint_loss_function(result: tuple[tf.Tensor], rtn_std: tf.Tensor, target_std: float, warm_up: int, alpha: float):
    """
    计算联合损失函数, 包括MLE损失和夏普比率损失。

    Args:
        result tuple[tf.Tensor]: 包含资产属性和仓位张量的 tuple。
        rtn_std (tf.Tensor): 资产的收益率和标准差张量。
        target_std (float): 目标标准差。
        warm_up (int): 暖启动阶段的时间步数。
        alpha (float): MLE损失在联合损失中的权重。

    Returns:
        tf.Tensor: 联合损失值。
    """
    properties, positions = result
    mle = alpha * mle_loss(properties, rtn_std, target_std, warm_up)
    sharpe = sharpe_loss(positions, rtn_std, target_std, warm_up)
    joint_loss = tf.reduce_mean(mle + sharpe)
    return joint_loss, tf.reduce_mean(mle), tf.reduce_mean(sharpe),


