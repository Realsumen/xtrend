import heartrate
import pandas as pd
from tqdm import tqdm

# heartrate.trace(browser=True)

data = pd.read_excel("data/C.CBT.xlsx")[["日期", "收盘价(元)"]]
data = data.rename(columns={"日期": "date", "收盘价(元)": "close"}).sort_values("date")

def get_MACD(data: pd.DataFrame, t: int, s: int, l: int) -> pd.DataFrame:
    """
    计算MACD指标。

    Args:
        data (pd.DataFrame): 包含日期和收盘价的数据。
        t (int): 时间窗口，用于计算滚动标准差。
        s (int): 短期指数平滑移动平均线(EMA)的平滑参数。
        l (int): 长期指数平滑移动平均线(EMA)的平滑参数。

    Returns:
        pd.DataFrame: 包含日期和MACD值的数据。
    """
    if data.isnull().any().any() or data.isna().any().any():
        raise ValueError("数据中存在 null 或者 na, 检查输入的数据")
    
    close_series, date_series = data["close"], data["date"]
    m = []
    for i in tqdm(range(60, len(data))):
        series = close_series[i - 60:i + t + 1]
        std = series.std()
        exp = lambda series, alpha: series.ewm(alpha=alpha, min_periods=alpha).mean().iloc[-1]
        ewma_s = exp(series, 1 / s)
        ewma_l = exp(series, 1 / l)
        m_t = (ewma_s - ewma_l) / std
        m.append(m_t)

    m = pd.Series(m, index=date_series[60:])
    print(m)
    m_std = m.rolling(t + 252).std()
    data["macd"] = m / m_std
    return data
    
# def get_r_hat():
    
get_MACD(data, t=21, s=8, l=24)
    