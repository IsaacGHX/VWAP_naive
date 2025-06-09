import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import timedelta

# --- 导入策略和模型 ---
from utils import generate_c53_signals
try:
    from dataloader_setup import FinancialDataset
    from VAE_trainer import TransformerVAE_TDist
    from vis import LatentOHLCVPredictor
except ImportError as e:
    st.error(f"导入模块失败: {e}。")
    st.stop()

from vwap_front import load_vae_model, load_predictor_model, load_financial_data


# --- 核心回测逻辑：V5 - 独立执行窗口 ---
def run_adaptive_backtest(predictor_model, dataset, raw_df, day_start_idx, model_seq_length, execution_window,
                          total_quantity, trade_direction, failsafe_ratio, device):
    """
    执行动态自适应回测策略。
    **V5版: 增加了独立的执行窗口，与模型输入序列长度解耦。**
    """
    remaining_quantity = float(total_quantity)
    results_log = []
    progress_bar = st.progress(0, text=f"正在执行 {execution_window} 分钟的VWAP策略...")

    # 主循环现在只运行用户定义的、更短的 execution_window
    for t in range(execution_window):
        order_quantity_for_this_minute = 0.0
        logic_used = ""
        predicted_close_for_this_minute = np.nan

        # --- 边界条件1: 提前完成 ---
        if remaining_quantity <= 1e-6:
            logic_used = "已完成"
            order_quantity_for_this_minute = 0.0

        # --- 边界条件2: 收盘冲刺 (基于执行窗口的最后5分钟) ---
        elif t >= execution_window - 5:
            logic_used = f"收盘冲刺 (r={failsafe_ratio})"
            minutes_left = execution_window - t
            if minutes_left == 1:
                order_quantity_for_this_minute = remaining_quantity
            else:
                # 处理 failsafe_ratio 为 1 的特殊情况以避免除零
                denominator = 1 - (failsafe_ratio ** minutes_left)
                if abs(denominator) < 1e-9:
                    order_quantity_for_this_minute = remaining_quantity / minutes_left
                else:
                    first_term = remaining_quantity * (1 - failsafe_ratio) / denominator
                    order_quantity_for_this_minute = first_term

        # --- 主要逻辑: 模型预测 ---
        else:
            logic_used = "模型预测"
            # 模型输入依然使用其固定的序列长度 model_seq_length
            input_start_idx = day_start_idx + t - model_seq_length
            input_end_idx = day_start_idx + t

            if input_start_idx >= 0:
                input_latent_seq_np = dataset.normalized_latent[input_start_idx:input_end_idx]
                input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

                with torch.no_grad():
                    # 模型仍然输出其固定长度 (model_seq_length) 的预测
                    outputs = predictor_model(input_tensor)['ohlcv_pred'].squeeze(0).cpu().numpy()
                    preds_scaled_back = dataset.ohlcv_scaler.inverse_transform(outputs)
                    preds_raw = np.expm1(preds_scaled_back)

                    # 【关键修正】我们只关心执行窗口内的预测
                    minutes_left_in_window = execution_window - t

                    # 从完整的模型预测中，只截取我们执行窗口剩余时间内的部分
                    relevant_future_preds = preds_raw[:minutes_left_in_window]

                    # 预测的成交量
                    predicted_volumes_in_window = relevant_future_preds[:, 5]  # Volume is at index 5

                    # 【关键修正】当前分钟的预测成交量是截取部分的第一个值
                    predicted_volume_for_now = predicted_volumes_in_window[0]

                    # 【关键修正】只对执行窗口内剩余时间的预测成交量求和
                    sum_of_future_predicted_volumes = np.sum(predicted_volumes_in_window)

                    # 预测的收盘价是截取部分的第一个预测收盘价
                    predicted_close_for_this_minute = relevant_future_preds[0, 3]  # Close is at index 3

                    if sum_of_future_predicted_volumes > 1e-6:
                        weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
                        order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
                    else:
                        # 备用逻辑：若预测成交量总和过小，则均分
                        order_quantity_for_this_minute = remaining_quantity / minutes_left_in_window
            else:
                # 若历史数据不足以进行预测，则在剩余时间内均分下单
                order_quantity_for_this_minute = remaining_quantity / (execution_window - t)

        # --- 执行与记录 (这部分逻辑不变) ---
        final_order_quantity = max(0.0, min(remaining_quantity, order_quantity_for_this_minute))

        current_minute_raw_data = raw_df.iloc[day_start_idx + t]
        execution_price = (current_minute_raw_data['high'] + current_minute_raw_data['low'] + current_minute_raw_data[
            'close']) / 3.0

        remaining_quantity -= final_order_quantity

        results_log.append({
            'timestamp': current_minute_raw_data['date'],
            'logic_used': logic_used,
            'predicted_price': predicted_close_for_this_minute,
            'execution_price': execution_price,
            'actual_volume': current_minute_raw_data['volume'],
            'order_quantity': final_order_quantity,
            'trade_value': final_order_quantity * execution_price,
            'remaining_quantity': remaining_quantity
        })
        progress_bar.progress((t + 1) / execution_window, text=f"执行中: {t + 1}/{execution_window} 分钟")

    # --- 后处理与指标计算 (这部分逻辑不变，但其作用域已变为 execution_window) ---
    results_df = pd.DataFrame(results_log)
    if results_df.empty or results_df['order_quantity'].sum() < 1e-6:
        return pd.DataFrame(), {}

    # (指标计算的剩余部分无需修改)
    total_actual_value_for_benchmark = (results_df['execution_price'] * results_df['actual_volume']).sum()
    total_actual_volume_for_benchmark = results_df['actual_volume'].sum()
    benchmark_vwap = total_actual_value_for_benchmark / total_actual_volume_for_benchmark if total_actual_volume_for_benchmark > 0 else 0
    model_total_trade_value = results_df['trade_value'].sum()
    model_total_quantity_traded = results_df['order_quantity'].sum()
    model_achieved_price = model_total_trade_value / model_total_quantity_traded if model_total_quantity_traded > 0 else 0
    if trade_direction.lower() == 'buy':
        slippage_per_share = benchmark_vwap - model_achieved_price
    else:
        slippage_per_share = model_achieved_price - benchmark_vwap
    total_cost_savings = slippage_per_share * total_quantity
    slippage_bps = (slippage_per_share / benchmark_vwap) * 10000 if benchmark_vwap > 0 else 0
    results_df['cumulative_benchmark_value'] = (results_df['execution_price'] * results_df['actual_volume']).cumsum()
    results_df['cumulative_actual_volume'] = results_df['actual_volume'].cumsum()
    results_df['traditional_vwap_line'] = results_df['cumulative_benchmark_value'] / results_df[
        'cumulative_actual_volume']
    results_df['cumulative_model_value'] = results_df['trade_value'].cumsum()
    results_df['cumulative_model_volume'] = results_df['order_quantity'].cumsum()
    results_df['model_vwap_line'] = (
                results_df['cumulative_model_value'] / results_df['cumulative_model_volume']).replace([np.inf, -np.inf],
                                                                                                      np.nan).ffill()
    metrics = {"Benchmark VWAP": benchmark_vwap, "Model Achieved Price": model_achieved_price,
               "Slippage Reduction (BPS)": slippage_bps, "Total Cost Savings": total_cost_savings}
    return results_df.set_index('timestamp'), metrics


# --- Streamlit UI ---
st.title("动态自适应VWAP策略回测平台 (V5 - 独立执行窗口版)")

st.info("""
**V5 更新:** VWAP执行逻辑与模型输入长度解耦。
- **执行窗口:** 您现在可以自定义VWAP策略的执行时长（例如15, 30, 60分钟），使其独立于模型所需的历史数据长度。
- **预测更精准:** 修正了预测逻辑，采用更标准的截取方式，使下单决策基于与执行窗口匹配的未来预测，提高了策略的相关性。
""")

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ 数据与模型设置")
    data_file = st.text_input("完整历史数据CSV路径", 'data/latent_features/c9999_1min_data.csv')
    st.subheader("模型路径")
    vae_model_path = st.text_input("VAE Model Path", 'best_tdist_vae_model.pth')
    predictor_model_path = st.text_input("Predictor Model Path", 'best_GPTpredictor_model.pth')

    st.header("🗓️ 回测设置")
    selected_date_str_placeholder = st.empty()

    # 【新增】信号周期控制参数
    signal_interval = st.selectbox(
        "信号周期 (分钟)",
        options=[1, 5, 15, 30, 60],
        index=1,  # 默认选中15分钟
        help="C53策略判断和生成信号的时间单位。例如，选择15，则会将1分钟数据合并为15分钟K线后再寻找交易机会。"
    )

    total_quantity = st.number_input("总交易数量", min_value=1, value=1000, step=100)

    # 【新增】让用户定义独立的VWAP执行窗口
    execution_window = st.number_input("VWAP执行窗口 (分钟)", min_value=10, max_value=240, value=30, step=5,
                                       help="信号触发后，执行订单的总分钟数。这是策略的核心时长。")

    st.header("📈 C53 策略参数")
    cl_param = st.slider("CL (通道周期)", 10, 100, 35)
    cd_param = st.slider("CD (通道偏移)", 0, 10, 0)
    stl_param = st.slider("STL (百分比止损 %)", 1.0, 10.0, 5.0, 0.5)
    n_param = st.slider("N (ATR止损倍数)", 1.0, 10.0, 6.0, 0.5)

    st.subheader("收盘冲刺参数")
    failsafe_ratio = st.slider("等比级数公比 (r)", 0.1, 1.0, 0.75, 0.05,
                               help="控制执行窗口最后5分钟的下单速度。")

    st.subheader("模型超参数")
    # 【修改】明确这是模型的固有参数，与策略执行时长分开
    model_seq_length = st.number_input("模型输入序列长度", value=345,
                                       help="这是预训练模型固有的输入序列长度，通常无需更改。")
    latent_dim = st.number_input("VAE Latent Dimension", value=16)
    ohlcv_dim = st.number_input("OHLCV Dimension", value=6)

    run_button = st.button("🚀 运行分析", type="primary", use_container_width=True)

# --- 主应用逻辑 ---
if data_file:
    try:
        temp_df = pd.read_csv(data_file, usecols=['date'])
        unique_dates = sorted(pd.to_datetime(temp_df['date']).dt.date.unique(), reverse=True)
        selected_date_str = selected_date_str_placeholder.selectbox("选择分析日期", options=unique_dates,
                                                                    key='selected_date_str')
    except Exception as e:
        st.error(f"加载日期失败: {e}")

if run_button and data_file:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: `{device}`")

    try:
        with st.spinner("加载并预处理数据..."):
            raw_df = pd.read_csv(data_file, parse_dates=['date'])
            # 【关键修改】数据重采样逻辑
            if signal_interval > 1:
                st.info(f"数据将按 {signal_interval} 分钟周期重采样以生成信号...")
                # 设置 'date' 列为索引以进行时间序列重采样
                resample_df = raw_df.set_index('date')

                # 定义聚合规则
                agg_rules = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }

                # 执行重采样
                signal_df_resampled = resample_df.resample(f'{signal_interval}T').agg(agg_rules).dropna()
                signal_df_resampled = signal_df_resampled.reset_index()
            else:
                # 如果周期为1分钟，则直接使用原始数据
                signal_df_resampled = raw_df.copy()

            # 后续的信号生成和分析都基于重采样后的数据
            dataset = load_financial_data(raw_df, data_file, model_seq_length, latent_dim)
            predictor_model = load_predictor_model(predictor_model_path, latent_dim, ohlcv_dim, model_seq_length,
                                                   device)

        with st.spinner(f"正在生成 {signal_interval} 分钟周期的C53信号..."):
            c53_params = {'cl_period': cl_param, 'cd_period': cd_param, 'stl_param': stl_param, 'n_param': n_param}
            # 在重采样后的DataFrame上生成信号
            signals_df = generate_c53_signals(signal_df_resampled, **c53_params)

        selected_date = pd.to_datetime(st.session_state.selected_date_str).date()
        # 从带有信号的(可能已重采样的)DataFrame中筛选当天数据
        day_df_signal = signals_df[signals_df['date'].dt.date == selected_date].copy()
        entry_signals = day_df_signal[(day_df_signal['signal'] == 'Buy') | (day_df_signal['signal'] == 'Sell')]

        if entry_signals.empty:
            st.warning(f"在 {selected_date} 的 {signal_interval} 分钟周期上没有找到C53开仓信号。")
            st.stop()

        first_signal = entry_signals.iloc[0]
        trade_direction = first_signal['signal']
        signal_time = first_signal['date']
        st.success(
            f"✅ 在 {signal_interval} 分钟周期上发现信号: **{trade_direction}** @ **{signal_time}**。开始在1分钟图上执行VWAP...")

        # 【重要】将信号时间点对齐回原始的1分钟K线，以启动VWAP回测
        # 我们使用 asof 在信号时间点找到原始1分钟数据中最接近的那一根K线
        signal_idx = raw_df.index[raw_df['date'] == pd.Timestamp(signal_time).as_unit('ns')].tolist()
        if not signal_idx:
            # 如果时间戳不能完美对齐，找到最近的
            signal_idx = [raw_df.index[raw_df['date'] <= signal_time][-1]]
        signal_idx = signal_idx[0]

        if signal_idx + execution_window > len(raw_df):
            st.error(
                f"数据不足：信号发生于 {signal_time}, 但其后没有足够的 {execution_window} 分钟1分钟数据来完成VWAP执行。")
            st.stop()

        # 【修改】调用更新后的回测函数，传入两个独立的长度参数
        results_df, metrics = run_adaptive_backtest(
            predictor_model, dataset, raw_df, signal_idx,
            model_seq_length,  # 模型的固有输入长度
            execution_window,  # 策略的实际执行时长
            total_quantity, trade_direction, failsafe_ratio, device
        )

        # (此处是上一部分的VWAP表现图代码)
        st.header(f"📈 当日 {signal_interval} 分钟K线与C53信号图")
        # 检查是否有当日数据可供绘图
        if not day_df_signal.empty:
            # 创建一个图表对象
            fig_kline = go.Figure()

            # 1. 添加K线图层
            fig_kline.add_trace(go.Candlestick(
                x=day_df_signal['date'], open=day_df_signal['open'], high=day_df_signal['high'],
                low=day_df_signal['low'], close=day_df_signal['close'], name=f'{signal_interval}分钟K线'
            ))

            # 2. 准备用于标记的信号数据
            buy_signals = day_df_signal[day_df_signal['signal'] == 'Buy']
            sell_signals = day_df_signal[day_df_signal['signal'] == 'Sell']
            close_buy_signals = day_df_signal[day_df_signal['signal'] == 'Close_Buy']
            close_sell_signals = day_df_signal[day_df_signal['signal'] == 'Close_Sell']

            # 3. 在图上添加信号标记
            # 使用向上的红色三角标记“开多”信号，放置在K线低点下方
            fig_kline.add_trace(go.Scatter(
                x=buy_signals['date'],
                y=buy_signals['low'] * 0.998,  # 乘以一个略小于1的数，使标记在K线下方
                mode='markers',
                marker=dict(color='red', symbol='triangle-up', size=12, line=dict(width=1, color='black')),
                name='开多 (Buy)'
            ))

            # 使用向下的绿色三角标记“开空”信号，放置在K线高点上方
            fig_kline.add_trace(go.Scatter(
                x=sell_signals['date'],
                y=sell_signals['high'] * 1.002,  # 乘以一个略大于1的数，使标记在K线上方
                mode='markers',
                marker=dict(color='green', symbol='triangle-down', size=12, line=dict(width=1, color='black')),
                name='开空 (Sell)'
            ))

            # 使用蓝色的'x'标记“平多”信号
            fig_kline.add_trace(go.Scatter(
                x=close_buy_signals['date'],
                y=close_buy_signals['close'],
                mode='markers',
                marker=dict(color='blue', symbol='x', size=10),
                name='平多 (Close Buy)'
            ))

            # 使用紫色的'x'标记“平空”信号
            fig_kline.add_trace(go.Scatter(
                x=close_sell_signals['date'],
                y=close_sell_signals['close'],
                mode='markers',
                marker=dict(color='purple', symbol='x', size=10),
                name='平空 (Close Sell)'
            ))

            # 4. 更新图表布局
            fig_kline.update_layout(
                title=f'{selected_date} 分钟K线与C53交易信号',
                xaxis_title='时间',
                yaxis_title='价格',
                xaxis_rangeslider_visible=False,  # 隐藏下方的范围滑动条，使界面更整洁
                legend_title="图例"
            )
            st.plotly_chart(fig_kline, use_container_width=True)

        # (此处是下一部分的分钟级交易日志代码)

        # (结果展示部分的逻辑完全不变，可以直接复用)
        st.header(f"📊 回测结果: {selected_date} ({execution_window}分钟窗口)")
        if not metrics:
            st.warning("未能计算回测指标，可能是因为没有发生任何交易。")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("基准VWAP", f"${metrics['Benchmark VWAP']:.4f}")
            col2.metric("模型实现均价", f"${metrics['Model Achieved Price']:.4f}")
            col3.metric("BPS滑点节省", f"{metrics['Slippage Reduction (BPS)']:.2f} bps")
            col4.metric("总成本节省", f"${metrics['Total Cost Savings']:,.2f}")

        st.header("📈 VWAP 与价格预测表现图")
        if not results_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['execution_price'], name='实际成交均价 (HLC/3)',
                                     line=dict(color='darkorange')))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['predicted_price'], name='模型预测收盘价',
                                     line=dict(color='mediumseagreen', dash='dot')))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['traditional_vwap_line'], name='基准 VWAP',
                                     line=dict(color='royalblue', width=3)))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['model_vwap_line'], name='模型自适应VWAP',
                                     line=dict(color='crimson', width=2, dash='dash')))
            fig.update_layout(title=f"VWAP & 价格预测对比 ({trade_direction})", yaxis_title="价格",
                              hovermode="x unified", legend_title="指标")
            st.plotly_chart(fig, use_container_width=True)

        st.header("📋 分钟级交易日志 (含预测)")
        if not results_df.empty:
            display_cols = ['logic_used', 'predicted_price', 'execution_price', 'order_quantity', 'trade_value',
                            'remaining_quantity']
            st.dataframe(results_df[display_cols].style.format({
                'predicted_price': '{:.4f}', 'execution_price': '{:.4f}',
                'order_quantity': '{:.2f}', 'trade_value': '{:,.2f}', 'remaining_quantity': '{:.2f}'
            }))

    except FileNotFoundError as e:
        st.error(f"文件未找到: {e}. 请检查侧边栏中的路径。")
    except Exception as e:
        st.error(f"发生未知错误: {e}")
        import traceback

        st.code(traceback.format_exc())

elif run_button:
    st.warning("⚠️ 请先在侧边栏指定历史数据CSV文件路径。")