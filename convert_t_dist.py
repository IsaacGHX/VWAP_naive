import pandas as pd
import torch
from VAE_trainer import TransformerVAE_TDist, preprocess_minute_data  # Assuming VAE_trainer.py is in the same directory
import numpy as np

# Define file paths
INPUT_CSV = r"data\raw_data\c9999_1min_data.csv"
OUTPUT_CSV = 'data\latent_features/c9999_1min_data.csv'
MODEL_PATH = 'best_tdist_vae_model.pth'  # Path to your trained model

# VAE model parameters (must match the parameters used for training)
FEATURE_DIM = 6  # open, high, low, close, volume, amount
LATENT_DIM = 16
EMBED_DIM = 64
# df parameter for the T-distribution VAE, must match training
DF_PARAM = 5.0

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def generate_latent_features(file_path, model_path, feature_dim, latent_dim, embed_dim, df_param, device):
    """
    Loads data, applies preprocessing, loads a trained VAE model,
    and generates latent features (z-samples) for the data.
    """
    # Load original data to preserve 'date' and 'position' columns
    original_df = pd.read_csv(file_path, parse_dates=['date'])

    # Preprocess the data using the function from VAE_trainer.py
    # This function returns a numpy array of preprocessed features
    processed_data_np = preprocess_minute_data(file_path, num_features=feature_dim)
    processed_data_tensor = torch.from_numpy(processed_data_np).float().to(device)

    # Initialize the VAE model
    model = TransformerVAE_TDist(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        df=df_param
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    latent_features = []
    with torch.no_grad():
        # Process data in batches, similar to how DataLoader would
        # Assuming batch_size of 345 for daily data, as per your trainer
        batch_size = 345
        for i in range(0, processed_data_tensor.shape[0], batch_size):
            batch_data = processed_data_tensor[i:i + batch_size]

            # Encode to get the parameters of the latent distribution
            mu, log_scale, log_df = model.encode(batch_data)

            # Reparameterize to sample z
            z, _ = model.reparameterize(mu, log_scale, log_df)

            latent_features.append(z.cpu().numpy())

    # Concatenate all latent features
    latent_features_np = np.concatenate(latent_features, axis=0)

    # Align original DataFrame with processed data (due to filtering in preprocess_minute_data)
    # The preprocess_minute_data filters for days with exactly 345 minutes and no zero open prices.
    # We need to apply the same filtering to the original_df to ensure alignment.
    # print("Re-filtering original DataFrame for alignment...")
    # original_df['day'] = original_df['date'].dt.date
    # df_aligned = original_df.groupby('day').filter(lambda x: len(x) == 345)
    # daily_opens = df_aligned.groupby('day')['open'].transform('first')
    # if (daily_opens == 0).any():
    #     bad_days = df_aligned[daily_opens == 0]['day'].unique()
    #     df_aligned = df_aligned[~df_aligned['day'].isin(bad_days)]

    df_aligned = original_df
    # Ensure the number of rows matches after filtering
    if len(df_aligned) != latent_features_np.shape[0]:
        raise ValueError(f"Mismatch in number of rows: Original aligned DF has {len(df_aligned)} rows, "
                         f"Latent features have {latent_features_np.shape[0]} rows. "
                         f"This might be due to inconsistencies in data filtering or 'position' column not being included in feature_dim correctly.")

    # Create a DataFrame for latent features
    latent_df = pd.DataFrame(latent_features_np,
                             columns=[f'latent_dim_{i + 1}' for i in range(latent_dim)],
                             index=df_aligned.index)  # Use aligned index for merging

    # Concatenate original data with latent features
    # Ensure date and time information are preserved in the original dataframe and then align.
    final_df = pd.concat([df_aligned.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)

    # Drop the temporary 'day' column
    # final_df = final_df.drop(columns=['day'])

    return final_df


if __name__ == '__main__':
    try:
        final_dataframe = generate_latent_features(
            file_path=INPUT_CSV,
            model_path=MODEL_PATH,
            feature_dim=FEATURE_DIM,
            latent_dim=LATENT_DIM,
            embed_dim=EMBED_DIM,
            df_param=DF_PARAM,
            device=DEVICE
        )

        # Save the combined DataFrame to a new CSV file
        final_dataframe.to_csv(OUTPUT_CSV, index=False)
        print(f"Latent features successfully generated and saved to {OUTPUT_CSV}")
        print(f"Final DataFrame shape: {final_dataframe.shape}")
        print(final_dataframe.head())

    except Exception as e:
        print(f"An error occurred: {e}")