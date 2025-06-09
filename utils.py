import pandas_ta as ta

def generate_c53_signals(df, cl_period=35, cd_period=0, stl_param=5.0, n_param=6.0, long_ma=240, short_ma=26):
    """
    根据C53策略逻辑生成交易信号。
    此函数使用pandas重现backtrader策略，以便在Streamlit应用中集成。

    Args:
        df (pd.DataFrame): 包含'high', 'low', 'close'列的原始数据。
        cl_period (int): 通道周期 (C53中的CL)。
        cd_period (int): 通道偏移 (C53中的CD)。
        stl_param (float): 百分比止损 (C53中的STL)。
        n_param (float): ATR止损倍数 (C53中的N)。
        long_ma (int): 长期均线周期。
        short_ma (int): 用于ATR的短期均线周期。

    Returns:
        pd.DataFrame: 带有 'signal' 列的DataFrame。
    """
    if df.empty:
        return df

    # 1. 使用pandas_ta计算所需指标
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=short_ma)
    df['mas'] = ta.ema(df['close'], length=long_ma)
    df['upperc'] = df['high'].rolling(window=cl_period).max()
    df['lowerc'] = df['low'].rolling(window=cl_period).min()

    # 2. 为匹配backtrader逻辑，将指标值向后移动
    df['upperc_shifted'] = df['upperc'].shift(cd_period + 1)
    df['lowerc_shifted'] = df['lowerc'].shift(cd_period + 1)
    df['mas_shifted'] = df['mas'].shift(1)
    df['high_prev'] = df['high'].shift(1)
    df['low_prev'] = df['low'].shift(1)
    df['close_prev'] = df['close'].shift(1)

    # 3. 定义入场条件
    duo_condition = (df['high'] >= df['upperc_shifted']) & \
                    (df['high_prev'] < df['upperc_shifted']) & \
                    (df['close_prev'] > df['mas_shifted'])

    kong_condition = (df['low'] <= df['lowerc_shifted']) & \
                     (df['low_prev'] > df['lowerc_shifted']) & \
                     (df['close_prev'] < df['mas_shifted'])

    # 4. 使用状态机模拟持仓和止损
    signals = ['Hold'] * len(df)
    position = 0  # -1代表空仓, 0代表无仓位, 1代表多仓
    bkhigh = 0.0    # 多头入场后的最高价
    sklow = float('inf')  # 空头入场后的最低价

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        # --- 止损逻辑 ---
        if position == 1:  # 持有多仓
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:  # 持有空仓
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 (优先处理止损) ---
        if position == 0:
            if duo_condition.iloc[i]:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            elif kong_condition.iloc[i]:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df