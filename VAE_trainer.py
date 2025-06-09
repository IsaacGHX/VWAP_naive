import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def preprocess_minute_data(file_path, num_features=7):
    """
    加载并预处理分钟级数据。
    V2版: 增加了数据质量检查和最终特征分布的打印。
    """
    print("Loading data...")
    df = pd.read_csv(file_path, parse_dates=['date'])

    # Get initial unique days
    initial_days = set(df['date'].dt.date.unique())
    print(f"Initial unique days in dataset: {len(initial_days)}")

    # 创建一个 'day' 列用于分组
    df['day'] = df['date'].dt.date

    # # 1. 筛选出每日恰好有345分钟的数据
    # print("Filtering days with exactly 345 minutes...")
    # day_counts = df.groupby('day').size()
    # # Identify days that do not have 345 minutes
    # days_not_345_minutes = day_counts[day_counts != 345].index.tolist()
    # if days_not_345_minutes:
    #     print(
    #         f"⚠️ Warning: Found {len(days_not_345_minutes)} days not having exactly 345 minutes. These days will be removed.")
    #     for d in days_not_345_minutes:
    #         print(f"  - Day: {d}, Minutes found: {day_counts[d]}")
    #
    # df_filtered = df.groupby('day').filter(lambda x: len(x) == 345)
    #
    # if df_filtered.empty:
    #     raise ValueError("No days with exactly 345 minutes found. Please check your data.")

    df_filtered = df

    # 增加检查：确保每日开盘价不为0，避免除零错误
    daily_opens = df_filtered.groupby('day')['open'].transform('first')
    if (daily_opens == 0).any():
        print("⚠️ Warning: Found days with an opening price of 0. These days will be removed.")
        bad_days = df_filtered[daily_opens == 0]['day'].unique()
        print("Found bad days: ", bad_days)
        df_filtered = df_filtered[~df_filtered['day'].isin(bad_days)]
        # 重新计算daily_opens
        daily_opens = df_filtered.groupby('day')['open'].transform('first')

    print(f"Found {len(df_filtered['day'].unique())} valid trading days after all checks.")

    # 2. 特征工程与归一化
    print("Performing feature engineering and scaling...")

    # 价格归一化
    # df_filtered['open'] = (df_filtered['open'] / daily_opens) - 1
    # df_filtered['high'] = (df_filtered['high'] / daily_opens) - 1
    # df_filtered['low'] = (df_filtered['low'] / daily_opens) - 1
    # df_filtered['close'] = (df_filtered['close'] / daily_opens) - 1

    # 对数化 + 标准化
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    # for col in ['volume', 'amount', 'position']:
    for col in feature_columns:
        df_filtered[col] = np.log1p(df_filtered[col])
        scaler = StandardScaler()
        df_filtered[col] = scaler.fit_transform(df_filtered[[col]])

    processed_data = df_filtered[feature_columns].values.astype(np.float32)

    # --- 3. 数据诊断 ---
    print("\n--- Data Diagnostics ---")

    # 检查是否存在 NaN 或 Inf
    has_nan = np.isnan(processed_data).any()
    has_inf = np.isinf(processed_data).any()

    if has_nan or has_inf:
        print(f"❌ Error: Processed data contains invalid values!")
        print(f"  - Contains NaN: {has_nan}")
        print(f"  - Contains Inf: {has_inf}")
        # 如果有问题，可以进一步定位具体是哪一列
        if has_nan:
            print("Columns with NaN:",
                  [feature_columns[i] for i, col_has_nan in enumerate(np.isnan(processed_data).any(axis=0)) if
                   col_has_nan])
        if has_inf:
            print("Columns with Inf:",
                  [feature_columns[i] for i, col_has_inf in enumerate(np.isinf(processed_data).any(axis=0)) if
                   col_has_inf])

    else:
        print("✅ Success: Processed data is clean (No NaN or Inf values).")

    # 计算并显示每个特征的均值和方差
    means = np.mean(processed_data, axis=0)
    variances = np.var(processed_data, axis=0)

    print("\nStatistics of Processed Features:")
    print("-" * 40)
    print(f"{'Feature':<12} | {'Mean':>10} | {'Variance':>12}")
    print("-" * 40)
    for i, name in enumerate(feature_columns):
        print(f"{name:<12} | {means[i]:>10.4f} | {variances[i]:>12.4f}")
    print("-" * 40)

    print("\nData processing complete.")
    return processed_data


class TransformerVAE_TDist(nn.Module):
    def __init__(self, feature_dim, latent_dim, embed_dim=64, nhead=4, num_layers=3, df=5.0):
        super(TransformerVAE_TDist, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.df = df  # T分布的自由度参数

        # 编码器部分保持不变
        self.encoder_input_fc = nn.Linear(feature_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出T分布参数: mu, log_scale, log_df
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_log_scale = nn.Linear(embed_dim, latent_dim)
        self.fc_log_df = nn.Linear(embed_dim, latent_dim)

        # 解码器部分保持不变
        self.decoder_input_fc = nn.Linear(latent_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_output_fc = nn.Linear(embed_dim, feature_dim)

        # 初始化为接近目标df的值 (df=5 -> log_df≈1.0986)
        nn.init.constant_(self.fc_log_df.bias, np.log(df - 2))
        # 限制权重范围
        self.fc_log_df.weight.data.uniform_(-0.001, 0.001)

    def encode(self, x):
        x = x.unsqueeze(1)
        x = F.silu(self.encoder_input_fc(x))
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        mu = self.fc_mu(x)
        log_scale = self.fc_log_scale(x)
        log_df = self.fc_log_df(x)
        return mu, log_scale, log_df

    def reparameterize(self, mu, log_scale, log_df):
        """T分布的重参数化技巧"""
        scale = torch.exp(log_scale)
        df = 3 + 27 * torch.sigmoid(log_df)  # 3 + 27*(0~1)

        # 使用StudentT分布
        t_dist = torch.distributions.StudentT(
            df=df,
            loc=mu,
            scale=scale
        )

        # 重参数化采样
        z = t_dist.rsample()
        return z, (mu, log_scale, log_df)

    def decode(self, z):
        z = F.silu(self.decoder_input_fc(z))
        z = z.unsqueeze(1)
        output = self.transformer_decoder(tgt=z, memory=z)
        output = output.squeeze(1)
        recon_x = self.decoder_output_fc(output)
        return recon_x

    def forward(self, x):
        mu, log_scale, log_df = self.encode(x)
        z, params = self.reparameterize(mu, log_scale, log_df)
        recon_x = self.decode(z)
        return recon_x, params


import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


# 定义细粒度损失计算函数
def detailed_loss_function(recon_x, x):
    """
    计算各特征的独立重建损失
    返回: 字典包含各项指标
    """

    # 价格指标 (OHLC)
    price_cols = slice(0, 4)  # 前4列是OHLC
    price_mae = F.l1_loss(recon_x[:, price_cols], x[:, price_cols]).item()
    price_mse = F.mse_loss(recon_x[:, price_cols], x[:, price_cols]).item()

    # 成交量指标 (volume, amount, position)
    volume_cols = slice(4, 7)
    volume_mae = F.l1_loss(recon_x[:, volume_cols], x[:, volume_cols]).item()
    volume_mse = F.mse_loss(recon_x[:, volume_cols], x[:, volume_cols]).item()

    # 整体指标
    total_mae = F.l1_loss(recon_x, x).item()
    total_mse = F.mse_loss(recon_x, x).item()

    results = {
        'price_mae': price_mae,
        'price_mse': price_mse,
        'volume_mae': volume_mae,
        'volume_mse': volume_mse,
        'total_mae': total_mae,
        'total_mse': total_mse
    }

    return results


def vae_loss_function(recon_x, x, mu, log_var):
    """改进的VAE损失函数，区分价格和成交量"""
    # 价格部分使用Huber损失
    price_loss = F.huber_loss(recon_x[:, :4], x[:, :4], reduction='sum')

    # 成交量部分使用MSE
    volume_loss = F.mse_loss(recon_x[:, 4:], x[:, 4:], reduction='sum')

    # KL散度
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return price_loss + volume_loss + kl_div


def t_dist_kl_divergence(mu_q, log_scale_q, log_df_q, mu_p=0, scale_p=1, df_p=5.0, df_penalty_weight=0.1):
    """
    改进的KL散度计算，添加对自由度的显式约束
    df_penalty_weight: 控制对自由度的惩罚强度
    """
    scale_q = torch.exp(log_scale_q)
    df_q = torch.exp(log_df_q) + 2.0

    df_p = torch.tensor(df_p, dtype=torch.float)
    scale_p = torch.tensor(scale_p, dtype=torch.float)

    # 原始KL计算
    term1 = torch.lgamma((df_q + 1) / 2) - torch.lgamma(df_q / 2)
    term2 = torch.lgamma((df_p + 1) / 2) - torch.lgamma(df_p / 2)
    term3 = 0.5 * (torch.log(df_q) + torch.log(scale_q ** 2) - torch.log(df_p) - torch.log(scale_p ** 2))
    term4 = ((df_q + 1) / 2) * (
        torch.log1p(
            (mu_q ** 2 + scale_q ** 2 * (df_q / (df_q - 2)) - 2 * mu_q * mu_p + mu_p ** 2) / (df_q * scale_p ** 2))
    )
    term5 = ((df_p + 1) / 2) * (
        torch.log1p(
            (mu_q ** 2 + scale_p ** 2 * (df_p / (df_p - 2)) - 2 * mu_q * mu_p + mu_p ** 2) / (df_p * scale_p ** 2))
    )

    kl = term1 - term2 - term3 + term4 - term5

    # 添加自由度惩罚项 (鼓励df接近df_p)
    df_penalty = df_penalty_weight * F.mse_loss(log_df_q, torch.log(torch.tensor(df_p - 2.0).to(log_df_q.device)))

    return torch.sum(kl) + df_penalty


def vae_loss_function_tdist(recon_x, x, params):
    """T分布VAE的损失函数"""
    mu_q, log_scale_q, log_df_q = params

    # 重建损失 (区分价格和成交量)
    price_loss = F.huber_loss(recon_x[:, :4], x[:, :4], reduction='sum')
    volume_loss = F.mse_loss(recon_x[:, 4:], x[:, 4:], reduction='sum')
    recon_loss = price_loss + volume_loss

    # KL散度 (T分布)
    kl_div = t_dist_kl_divergence(mu_q, log_scale_q, log_df_q)

    return recon_loss + kl_div


def prepare_dataloaders(processed_data, batch_size=345, test_size=0.05):
    """准备训练和验证集的DataLoader"""
    # 按天数分割确保时间连续性
    n_days = len(processed_data) // 345
    train_size = int((1 - test_size) * n_days)

    # 按天分割
    train_data = processed_data[:train_size * 345]
    val_data = processed_data[train_size * 345:]

    train_dataset = TensorDataset(torch.from_numpy(train_data).float())
    val_dataset = TensorDataset(torch.from_numpy(val_data).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    """完整的训练流程"""
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }

        for data, in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(data)
            loss = vae_loss_function(recon_batch, data, mu, log_var)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = detailed_loss_function(recon_batch.detach(), data.detach())
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]

        # 计算平均训练指标
        train_loss /= len(train_loader.dataset)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }

        with torch.no_grad():
            for data, in val_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                val_loss += vae_loss_function(recon_batch, data, mu, log_var).item()

                batch_metrics = detailed_loss_function(recon_batch, data)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]

        # 计算平均验证指标
        val_loss /= len(val_loader.dataset)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

        # 打印进度
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Train Metrics:")
        print(f"  Price - MAE: {train_metrics['price_mae']:.4f}, MSE: {train_metrics['price_mse']:.4f}")
        print(f"  Volume - MAE: {train_metrics['volume_mae']:.4f}, MSE: {train_metrics['volume_mse']:.4f}")
        print("Val Metrics:")
        print(f"  Price - MAE: {val_metrics['price_mae']:.4f}, MSE: {val_metrics['price_mse']:.4f}")
        print(f"  Volume - MAE: {val_metrics['volume_mae']:.4f}, MSE: {val_metrics['volume_mse']:.4f}")

        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print("Saved new best model")

    return history


def train_model_tdist(model, train_loader, val_loader, optimizer, epochs, device):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }

        for data, in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, params = model(data)
            loss = vae_loss_function_tdist(recon_batch, data, params)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = detailed_loss_function(recon_batch.detach(), data.detach())
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]

        # 验证阶段
        model.eval()
        val_loss = 0
        val_metrics = {
            'price_mae': 0, 'price_mse': 0,
            'volume_mae': 0, 'volume_mse': 0
        }

        with torch.no_grad():

            sample = next(iter(val_loader))[0].to(DEVICE)
            _, _, log_df = model.encode(sample)
            avg_df = torch.exp(log_df).mean().item() + 2
            print(f"Average learned degrees of freedom: {avg_df:.2f}")

            for data, in val_loader:
                data = data.to(device)
                recon_batch, params = model(data)
                val_loss += vae_loss_function_tdist(recon_batch, data, params).item()

                batch_metrics = detailed_loss_function(recon_batch, data)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]

        # 记录和打印
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            val_metrics[k] /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Train Metrics:")
        print(f"  Price - MAE: {train_metrics['price_mae']:.4f}, MSE: {train_metrics['price_mse']:.4f}")
        print(f"  Volume - MAE: {train_metrics['volume_mae']:.4f}, MSE: {train_metrics['volume_mse']:.4f}")

        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_tdist_vae_model.pth')
            print("Saved new best model")

    return history

def plot_training_history(history):
    """绘制训练过程图表"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 价格MAE
    plt.subplot(2, 2, 2)
    train_price_mae = [m['price_mae'] for m in history['train_metrics']]
    val_price_mae = [m['price_mae'] for m in history['val_metrics']]
    plt.plot(train_price_mae, label='Train Price MAE')
    plt.plot(val_price_mae, label='Val Price MAE')
    plt.title('Price MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 成交量MAE
    plt.subplot(2, 2, 3)
    train_volume_mae = [m['volume_mae'] for m in history['train_metrics']]
    val_volume_mae = [m['volume_mae'] for m in history['val_metrics']]
    plt.plot(train_volume_mae, label='Train Volume MAE')
    plt.plot(val_volume_mae, label='Val Volume MAE')
    plt.title('Volume MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 价格MSE
    plt.subplot(2, 2, 4)
    train_price_mse = [m['price_mse'] for m in history['train_metrics']]
    val_price_mse = [m['price_mse'] for m in history['val_metrics']]
    plt.plot(train_price_mse, label='Train Price MSE')
    plt.plot(val_price_mse, label='Val Price MSE')
    plt.title('Price MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


if __name__ == '__main__':
    # 参数设置
    FILE_PATH = 'data/raw_data/c9999_1min_data.csv'
    FEATURE_DIM = 6
    LATENT_DIM = 16
    EMBED_DIM = 64
    EPOCHS = 50
    BATCH_SIZE = 345
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # 数据准备
    try:
        processed_data = preprocess_minute_data(FILE_PATH, num_features=FEATURE_DIM)
        train_loader, val_loader = prepare_dataloaders(processed_data, BATCH_SIZE)
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()

    # 初始化T分布VAE模型
    model = TransformerVAE_TDist(
        feature_dim=FEATURE_DIM,
        latent_dim=LATENT_DIM,
        embed_dim=EMBED_DIM,
        df=5.0  # 初始自由度
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    history = train_model_tdist(
        model, train_loader, val_loader,
        optimizer, EPOCHS, DEVICE
    )

    # 可视化结果
    plot_training_history(history)

    print("Training completed. Best T-distribution VAE saved to 'best_tdist_vae_model.pth'")