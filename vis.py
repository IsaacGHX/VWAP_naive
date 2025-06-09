import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader_setup import LatentOHLCVPredictor, FinancialDataset
from VAE_trainer import TransformerVAE_TDist
from  tqdm import tqdm

def visualize_last_n_days(predictor_model, vae_model, dataset, latent_scaler, ohlcv_scaler, n_days=5, seq_length=345,
                          device='cpu'):
    """
    Generates predictions for the last n days and plots them against original data.
    """
    print(f"\n--- Starting Generation for Last {n_days} Days ---")
    predictor_model.eval()
    vae_model.eval()

    total_minutes = len(dataset.df)

    if total_minutes < (n_days + 1) * seq_length:
        print(f"Error: Not enough data. Need at least {(n_days + 1) * seq_length} minutes, but found {total_minutes}.")
        return

    for i in range(1, n_days + 1):
        # --- 1. Get Input: Latent sequence from the PREVIOUS day ---
        # The day before the target day. e.g., for the last day (i=1), use the second-to-last day as input.
        start_idx_input = total_minutes - (i + 1) * seq_length
        end_idx_input = total_minutes - i * seq_length

        print(f"\nGenerating for target day starting at minute index {end_idx_input}...")
        print(f"Using input data from minute indices {start_idx_input} to {end_idx_input - 1}")

        initial_latent_seq_np = dataset.normalized_latent[start_idx_input:end_idx_input]
        initial_latent_for_gen = torch.FloatTensor(initial_latent_seq_np).unsqueeze(0).to(device)

        # --- 2. Get Ground Truth: Raw OHLCV from the TARGET day ---
        start_idx_target = end_idx_input
        end_idx_target = start_idx_target + seq_length

        original_ohlcv_raw = dataset.ohlcv_features[start_idx_target:end_idx_target]
        original_date = dataset.dates[start_idx_target]

        # --- 3. Generate ---
        generated_output = predictor_model.generate(
            initial_latent_for_gen, vae_model, ohlcv_scaler, latent_scaler,
            steps=seq_length, device=device, use_cache=True
        )

        # --- 4. Process Generated Data (Inverse Transform) ---
        generated_ohlcv_normalized_np = generated_output['ohlcv'].squeeze(0).cpu().numpy()
        generated_ohlcv_scaled_back = ohlcv_scaler.inverse_transform(generated_ohlcv_normalized_np)
        generated_ohlcv_final_raw = np.expm1(generated_ohlcv_scaled_back)

        # --- 5. Visualize Comparison ---
        plt.figure(figsize=(15, 7))
        # Plotting the 'close' price (column index 3)
        plt.plot(original_ohlcv_raw[:, 4], label='Original Close Price', color='blue', alpha=0.8)
        plt.plot(generated_ohlcv_final_raw[:, 4], label='Generated Close Price', color='red', linestyle='--')

        plt.title(f'OHLCV Generation vs. Original Data\nTarget Date: {original_date}')
        plt.xlabel(f'Time Step (Minute of Day)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{original_date}".split(" ")[0])
        plt.show()

def visualize_last_n_days_single_step(predictor_model, dataset, ohlcv_scaler, n_days=5, seq_length=345, device='cpu'):
    """
    NEW FUNCTION: Generates single-step predictions for the last n days and plots them.
    Each prediction is based on the true historical data preceding it (rolling forecast).
    """
    print(f"\n--- Starting Single-Step Prediction for Last {n_days} Days ---")
    predictor_model.eval()

    total_minutes = len(dataset.df)

    # We need at least seq_length minutes of history before the first prediction point.
    if total_minutes < (n_days * seq_length) + seq_length:
        print(f"Error: Not enough data. Need at least {(n_days * seq_length) + seq_length} minutes, but found {total_minutes}.")
        return

    # Outer loop for each of the last N days
    for i in range(1, n_days + 1):
        day_start_index = total_minutes - i * seq_length
        day_end_index = day_start_index + seq_length

        original_ohlcv_raw = dataset.ohlcv_features[day_start_index:day_end_index]
        original_date = dataset.dates[day_start_index]

        print(f"\nPredicting for target day starting at minute index {day_start_index} (Date: {original_date})")

        all_single_step_preds_normalized = []

        # Inner loop for each minute within the target day
        for j in tqdm(range(seq_length), desc=f"Predicting Day T-{(n_days - i)}"):
            # Define the sliding window of historical data (seq_length minutes immediately preceding the target)
            start_idx_input = day_start_index + j - seq_length
            end_idx_input = day_start_index + j

            input_latent_seq_np = dataset.normalized_latent[start_idx_input:end_idx_input]
            input_tensor = torch.FloatTensor(input_latent_seq_np).unsqueeze(0).to(device)

            # --- Predict just one step forward using the standard forward pass ---
            with torch.no_grad():
                outputs = predictor_model(input_tensor)
                # The prediction for the next step is the last element in the output sequence
                next_step_pred_normalized = outputs['ohlcv_pred'][:, -1, :]  # Shape: [1, ohlcv_dim]

            all_single_step_preds_normalized.append(next_step_pred_normalized)

        # --- Process all collected predictions for the day ---
        generated_ohlcv_normalized = torch.cat(all_single_step_preds_normalized, dim=0)
        generated_ohlcv_normalized_np = generated_ohlcv_normalized.cpu().numpy()
        generated_ohlcv_scaled_back = ohlcv_scaler.inverse_transform(generated_ohlcv_normalized_np)
        generated_ohlcv_final_raw = np.expm1(generated_ohlcv_scaled_back)

        # --- Visualize Comparison ---
        plt.figure(figsize=(15, 7))
        # Plotting the 'close' price (column index 3)
        plt.plot(original_ohlcv_raw[:, 4], label='Original Close Price', color='blue', alpha=0.8)
        plt.plot(generated_ohlcv_final_raw[:, 4], label='Single-Step Predicted Close Price', color='green', linestyle='--')

        plt.title(f'Single-Step OHLCV Prediction vs. Original Data\nTarget Date: {original_date}')
        plt.xlabel(f'Time Step (Minute of Day)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{original_date}".split(" ")[0])
        plt.show()

def main():
    """
    Main function to load models and run the visualization task.
    """
    # --- Configuration ---
    latent_csv_path = "data/latent_features/c9999_1min_data.csv"
    vae_model_path = 'best_tdist_vae_model.pth'
    predictor_model_path = 'best_GPTpredictor_model.pth'  # Path to your saved predictor model

    seq_length = 345
    vae_latent_dim = 16
    ohlcv_dim = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    print("Loading and preparing dataset...")
    try:
        full_dataset = FinancialDataset(
            latent_csv_path=latent_csv_path,
            seq_length=seq_length,
            latent_dim=vae_latent_dim
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {latent_csv_path}. Please check the path.")
        return
    print("Dataset loaded successfully.")

    # --- Load VAE Model ---
    vae_model = TransformerVAE_TDist(
        feature_dim=ohlcv_dim, latent_dim=vae_latent_dim, embed_dim=64, df=5.0
    ).to(device)
    try:
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=device))
        vae_model.eval()
        print(f"Successfully loaded VAE model from {vae_model_path}")
    except FileNotFoundError:
        print(f"Error: VAE model not found at {vae_model_path}. Please ensure it is trained and the path is correct.")
        return

    # --- Load Predictor Model ---
    model = LatentOHLCVPredictor(
        latent_dim=vae_latent_dim, ohlcv_dim=ohlcv_dim, seq_length=seq_length
    ).to(device)
    try:
        model.load_state_dict(torch.load(predictor_model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded Predictor model from {predictor_model_path}")
    except FileNotFoundError:
        print(
            f"Error: Predictor model not found at {predictor_model_path}. Please ensure it is trained and the path is correct.")
        return

    # --- Run Visualization ---
    # visualize_last_n_days(
    #     predictor_model=model,
    #     vae_model=vae_model,
    #     dataset=full_dataset,
    #     latent_scaler=full_dataset.latent_scaler,
    #     ohlcv_scaler=full_dataset.ohlcv_scaler,
    #     n_days=5,
    #     seq_length=seq_length,
    #     device=device
    # )

    # Option 2: Single-Step Rolling Forecast (New method as requested)
    visualize_last_n_days_single_step(
        predictor_model=model,
        dataset=full_dataset,
        ohlcv_scaler=full_dataset.ohlcv_scaler,
        n_days=14,
        seq_length=seq_length,
        device=device
    )


if __name__ == "__main__":
    main()