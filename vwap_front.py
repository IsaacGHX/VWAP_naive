import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from tqdm import tqdm

# --- 导入模块 (无变化) ---
try:
    from dataloader_setup import FinancialDataset
    from VAE_trainer import TransformerVAE_TDist
    from vis import LatentOHLCVPredictor
except ImportError as e:
    st.error(f"导入模块失败: {e}. 请确保相关文件在同一目录下。")
    st.stop()


# --- 模型与数据加载函数 (无变化) ---
@st.cache_resource
def load_vae_model(vae_path, feature_dim, latent_dim, embed_dim, df, device):
    model = TransformerVAE_TDist(feature_dim=feature_dim, latent_dim=latent_dim, embed_dim=embed_dim, df=df).to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


@st.cache_resource
def load_predictor_model(predictor_path, latent_dim, ohlcv_dim, seq_length, device):
    model = LatentOHLCVPredictor(latent_dim=latent_dim, ohlcv_dim=ohlcv_dim, seq_length=seq_length).to(device)
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.eval()
    return model


@st.cache_data
def load_financial_data(_dataset, csv_path, seq_length, latent_dim):
    # _dataset 参数是为了让 Streamlit 的缓存知道数据文件已更新
    dataset = FinancialDataset(latent_csv_path=csv_path, seq_length=seq_length, latent_dim=latent_dim)
    return dataset


# --- 核心回测逻辑：修正价格获取方式 ---
def run_adaptive_backtest(predictor_model, dataset, raw_df, day_start_idx, seq_length,
                          total_quantity, trade_direction, failsafe_ratio, device):
    """
    执行动态自适应回测策略。
    **V3版: 修正了价格逻辑，使用原始HLC均价作为成交价。**
    """
    remaining_quantity = float(total_quantity)
    results_log = []

    for t in tqdm(range(seq_length), desc="Running Adaptive Backtest"):
        order_quantity_for_this_minute = 0.0
        logic_used = ""

        # --- 边界条件1: 提前完成 ---
        if remaining_quantity <= 1e-6:
            logic_used = "已完成"
            order_quantity_for_this_minute = 0.0
        # --- 边界条件2: 收盘冲刺 ---
        elif t >= seq_length - 5:
            logic_used = f"收盘冲刺 (r={failsafe_ratio})"
            minutes_left = seq_length - t
            if minutes_left == 1:
                order_quantity_for_this_minute = remaining_quantity
            else:
                if abs(1.0 - failsafe_ratio) < 1e-9:
                    first_term = remaining_quantity / minutes_left
                else:
                    first_term = remaining_quantity * (1 - failsafe_ratio) / (1 - failsafe_ratio ** minutes_left)
                order_quantity_for_this_minute = first_term
        # --- 主要逻辑: 模型预测 ---
        else:
            logic_used = "模型预测"
            input_start_idx = day_start_idx + t - seq_length
            input_end_idx = day_start_idx + t

            if input_start_idx < 0:
                order_quantity_for_this_minute = 0.0
            else:
                input_latent_seq_np = dataset.normalized_latent[input_start_idx:input_end_idx]
                input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = predictor_model(input_tensor)['ohlcv_pred'].squeeze(0).cpu().numpy()
                    preds_scaled_back = dataset.ohlcv_scaler.inverse_transform(outputs)
                    preds_raw = np.expm1(preds_scaled_back)
                    predicted_volumes_future = preds_raw[:, 5]

                    predicted_volume_for_now = predicted_volumes_future[-1]
                    sum_of_future_predicted_volumes = np.sum(predicted_volumes_future)

                    if sum_of_future_predicted_volumes > 1e-6:
                        weight_for_this_minute = predicted_volume_for_now / sum_of_future_predicted_volumes
                        order_quantity_for_this_minute = remaining_quantity * weight_for_this_minute
                    else:
                        minutes_left_in_model_horizon = seq_length - t
                        order_quantity_for_this_minute = remaining_quantity / minutes_left_in_model_horizon

        # --- 执行与记录 ---
        final_order_quantity = max(0.0, min(remaining_quantity, order_quantity_for_this_minute))

        # 【关键修正】从原始DataFrame获取未处理的价格和成交量
        current_minute_raw_data = raw_df.iloc[day_start_idx + t]
        high = current_minute_raw_data['high']
        low = current_minute_raw_data['low']
        close = current_minute_raw_data['close']
        actual_volume = current_minute_raw_data['volume']

        # 【关键修正】定义本分钟的成交价格
        execution_price = (high + low + close) / 3.0

        remaining_quantity -= final_order_quantity

        results_log.append({
            'timestamp': current_minute_raw_data['date'],
            'logic_used': logic_used,
            'execution_price': execution_price,  # 使用修正后的价格
            'actual_volume': actual_volume,
            'order_quantity': final_order_quantity,
            'trade_value': final_order_quantity * execution_price,  # 使用修正后的价格
            'remaining_quantity': remaining_quantity
        })

    # --- 后处理与指标计算 ---
    results_df = pd.DataFrame(results_log)
    if results_df.empty or results_df['order_quantity'].sum() < 1e-6:
        return pd.DataFrame(), {}

    # 【关键修正】基准VWAP的计算也使用统一的成交价标准 (execution_price)
    total_actual_value_for_benchmark = (results_df['execution_price'] * results_df['actual_volume']).sum()
    total_actual_volume_for_benchmark = results_df['actual_volume'].sum()
    benchmark_vwap = total_actual_value_for_benchmark / total_actual_volume_for_benchmark if total_actual_volume_for_benchmark > 0 else 0

    model_total_trade_value = results_df['trade_value'].sum()
    model_total_quantity_traded = results_df['order_quantity'].sum()
    model_achieved_price = model_total_trade_value / model_total_quantity_traded if model_total_quantity_traded > 0 else 0

    if trade_direction == 'Buy':
        slippage_per_share = benchmark_vwap - model_achieved_price
    else:  # Sell
        slippage_per_share = model_achieved_price - benchmark_vwap

    total_cost_savings = slippage_per_share * total_quantity
    slippage_bps = (slippage_per_share / benchmark_vwap) * 10000 if benchmark_vwap > 0 else 0

    # 计算绘图曲线
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
st.set_page_config(layout="wide")
st.title("动态自适应VWAP策略回测平台 (V3 - 价格修正版)")

st.info("""
**V3 更新:** 修正了核心的价格计算逻辑。
- **成交价格:** 每分钟的成交价不再使用模型内部的标准化数据，而是严格采用原始数据中的 **(最高价 + 最低价 + 收盘价) / 3**。
- **公平基准:** 用于计算基准VWAP的价格也采用上述同样标准，确保了策略与基准之间的对比是公平的。
""")

# ... 侧边栏UI部分与上一版完全相同 ...
with st.sidebar:
    st.header("⚙️ 模型与数据设置")
    data_file = st.text_input("包含OHLCV和特征的完整历史数据CSV路径", 'data/latent_features/c9999_1min_data.csv')
    st.subheader("模型路径")
    vae_model_path = st.text_input("VAE Model Path", 'best_tdist_vae_model.pth')
    predictor_model_path = st.text_input("Predictor Model Path", 'best_GPTpredictor_model.pth')

    st.subheader("回测参数")
    selected_date_str = st.selectbox("选择分析日期", options=[])
    trade_direction = st.selectbox("交易方向", options=['Buy', 'Sell'])
    total_quantity = st.number_input("总交易数量", min_value=1, value=10000, step=100)

    st.subheader("收盘冲刺参数")
    failsafe_ratio = st.slider("等比级数公比 (r)", 0.1, 1.0, 0.75, 0.05,
                               help="控制最后5分钟下单速度。r越小，下单量递减越快。")

    st.subheader("模型超参数")
    seq_length = st.number_input("Sequence Length", value=345)
    latent_dim = st.number_input("VAE Latent Dimension", value=16)
    ohlcv_dim = st.number_input("OHLCV Dimension", value=6)

    run_button = st.button("🚀 运行分析", type="primary", use_container_width=True)

# --- 主应用逻辑 ---
if data_file:
    # 动态更新日期选择器
    # 使用  (temporary) DataFrame 来避免重复加载完整数据
    temp_df = pd.read_csv(data_file, usecols=['date'])
    unique_dates = sorted(pd.to_datetime(temp_df['date']).dt.date.unique(), reverse=True)
    st.sidebar.selectbox("选择分析日期", options=unique_dates, key='selected_date_str')

if run_button and data_file:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: `{device}`")

    try:
        # 【关键修正】加载两份数据：一份给模型用，一份为原始数据用于回测
        with st.spinner("加载数据和模型中..."):
            # seek(0) 确保每次都能从文件开头读取
            raw_df = pd.read_csv(data_file, parse_dates=['date'])
            dataset = load_financial_data(raw_df, data_file, seq_length, latent_dim)

            vae_model = load_vae_model(vae_model_path, ohlcv_dim, latent_dim, 64, 5.0, device)
            predictor_model = load_predictor_model(predictor_model_path, latent_dim, ohlcv_dim, seq_length, device)

        # 找到日期的起始索引
        selected_date = pd.to_datetime(st.session_state.selected_date_str).date()
        # 我们需要从原始数据中定位，因为dataset.dates可能格式不同
        day_start_offset = raw_df.index[raw_df['date'].dt.date == selected_date]

        if len(day_start_offset) < seq_length:
            st.error(f"日期 {selected_date} 的数据不足 (需要 {seq_length} 分钟, 找到 {len(day_start_offset)} 分钟)。")
            st.stop()
        day_start_idx = day_start_offset[0]

        # 调用新的核心回测函数，并传入原始数据 raw_df
        results_df, metrics = run_adaptive_backtest(
            predictor_model, dataset, raw_df, day_start_idx, seq_length,
            total_quantity, trade_direction, failsafe_ratio, device
        )

        st.header(f"📊 动态自适应策略回测结果: {selected_date}")
        if not metrics:
            st.warning("未能计算回测指标，可能是因为没有发生任何交易。")
        else:
            # 指标展示
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("基准VWAP", f"${metrics['Benchmark VWAP']:.4f}")
            col2.metric("模型实现均价", f"${metrics['Model Achieved Price']:.4f}")
            col3.metric("BPS滑点节省", f"{metrics['Slippage Reduction (BPS)']:.2f} bps")
            col4.metric("总成本节省", f"${metrics['Total Cost Savings']:,.2f}")

        st.header("💰 VWAP 表现图")
        # 绘图...
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['traditional_vwap_line'], name='基准 VWAP (HLC/3)',
                                 line=dict(color='royalblue', width=3)))
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['model_vwap_line'], name='模型自适应VWAP',
                                 line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.header("📋 分钟级交易日志")
        # 详细日志展示...
        display_cols = ['logic_used', 'execution_price', 'order_quantity', 'trade_value', 'remaining_quantity']
        st.dataframe(results_df[display_cols].style.format(precision=2))

    except FileNotFoundError as e:
        st.error(f"模型文件未找到: {e}. 请检查侧边栏中的路径。")
    except Exception as e:
        st.error(f"发生错误: {e}")
        import traceback

        st.code(traceback.format_exc())

elif run_button:
    st.warning("⚠️ 请先上传您的历史数据CSV文件。")