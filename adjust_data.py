import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def detect_and_adjust_rollover_gaps(input_file, output_dir, output_file):
    """
    读取期货日线数据，检测并使用后复权方法修正因换月引起的跳空。

    :param input_file: 输入的CSV文件路径 (e.g., 'IF9999_daily_data.csv')
    :param output_file: 输出的复权后的CSV文件路径
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)

    # --- 1. 数据准备 ---
    # 确保date列是datetime类型，并按日期排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 创建一个副本用于复权操作，保留原始数据用于对比
    df_adjusted = df.copy()

    # --- 2. 计算检测指标 ---
    # 计算前一天的收盘价
    df_adjusted['prev_close'] = df_adjusted['close'].shift(1)

    # 计算价格跳空百分比
    df_adjusted['price_gap_pct'] = (df_adjusted['open'] - df_adjusted['prev_close']) / df_adjusted['prev_close']

    # 计算持仓量变化百分比
    df_adjusted['position_change_pct'] = df_adjusted['position'].pct_change()

    # --- 3. 识别换月点 ---
    # 定义阈值，这些值可能需要根据不同品种进行微调
    PRICE_GAP_THRESHOLD = 0.01  # 价格跳空阈值，例如 1.5%
    POSITION_CHANGE_THRESHOLD = 0.15  # 持仓量变化阈值，例如 10%

    # 识别条件：价格跳空和持仓量变化同时超过阈值
    rollover_mask = (df_adjusted['price_gap_pct'].abs() > PRICE_GAP_THRESHOLD) & \
                    (df_adjusted['position_change_pct'].abs() > POSITION_CHANGE_THRESHOLD)

    rollover_dates = df_adjusted[rollover_mask]
    print(f"\n检测到 {len(rollover_dates)} 个可能的换月点:")
    print(rollover_dates[['date', 'price_gap_pct', 'position_change_pct']])

    # --- 4. 执行后复权 ---
    # 从后往前遍历所有换月点，进行修正
    for idx in rollover_dates.index[::-1]:
        # 获取价差 (开盘价 - 前一天收盘价)
        gap = df_adjusted.loc[idx, 'open'] - df_adjusted.loc[idx, 'prev_close']

        print(f"\n正在修正 {df_adjusted.loc[idx, 'date'].date()} 的跳空，价差: {gap:.2f}")

        # 将此换月点之前的所有OHLC价格减去价差
        ohlc_cols = ['open', 'high', 'low', 'close']
        df_adjusted.loc[:idx - 1, ohlc_cols] -= gap

        print(f"已修正该日期之前的 {idx} 条数据。")

    # --- 5. 可视化对比 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(df['date'], df['close'], label='(Original Close)', color='skyblue', linewidth=1.5)
    ax.plot(df_adjusted['date'], df_adjusted['close'], label='(Adjusted Close)', color='crimson',
            linewidth=1.5)

    # 标记出换月点
    for date in rollover_dates['date']:
        ax.axvline(x=date, color='gray', linestyle='--', linewidth=0.8, label='_nolegend_')

    ax.set_title(f'{input_file.split("/")[-1]} after adjustment', fontsize=16)
    ax.set_xlabel('(Date)', fontsize=12)
    ax.set_ylabel('(Price)', fontsize=12)
    ax.legend()
    ax.grid(True)

    # 格式化日期显示
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plot_filename = f'_adjustment_plot.png'
    plt.savefig(plot_filename)
    print(f"\n📈 对比图已保存为: {plot_filename}")

    # --- 6. 保存复权后的数据 ---
    # 删除用于计算的临时列
    final_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'position']
    df_adjusted = df_adjusted[final_cols]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)

    df_adjusted.to_csv(output_file, index=False)
    print(f"✅ 复权后的数据已保存至: {output_file}")


# --- 主程序入口 ---
if __name__ == "__main__":


    # 1. 输入文件名 (上一步生成的CSV文件)
    INPUT_FILENAME = './data/raw_data/c9999_daily_data.csv'

    # 2. 输出文件名 (复权修正后的文件)
    OUTPUT_FILENAME = f'{INPUT_FILENAME.split("/")[-1]}_adjusted.csv'
    OUTPUT_DIR = "./data/adjusted_data"

    # --- 执行函数 ---
    detect_and_adjust_rollover_gaps(
        input_file=INPUT_FILENAME,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILENAME
    )