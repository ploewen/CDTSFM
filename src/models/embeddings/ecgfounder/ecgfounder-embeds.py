# Purpose:
# Makes ECGFounder embeddings for S&P 500, ESC 50, PTB XL Datasets.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py
# - Requires running ecgfounder-setup.sh

# Authors:
# - Code written by Philip Loewen

# References:
# J. Li et al., “An Electrocardiogram Foundation Model Built on over 10 Million Recordings,”
# NEJM AI, vol. 2, no. 7, Jun. 2025, doi: 10.1056/aioa2401033.

import torch
import pandas as pd
from tqdm import tqdm
import os
import sys

# Import model
try:
    from net1d import Net1D
except ImportError:
    print("Error: 'net1d.py' not found. Please ensure the file exists.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# # Get the path to weights
WEIGHTS_PATH = "weights/1_lead_ECGFounder.pth"


def load_1ch_model(weights_path):
    """
    Initializes and loads the 1-Channel ECGFounder model.

    This function instantiates the Net1D architecture with specific hyperparameters
    designed for 1-channel input (modified from the original 12-lead version).
    It handles loading the state dictionary, stripping 'module.' prefixes if present,
    and sets the model to evaluation mode.

    Args:
        weights_path (str): Path to the .pth file containing pre-trained weights.

    Returns:
        torch.nn.Module: The loaded Net1D model on the configured device.

    Raises:
        SystemExit: If the weights file cannot be loaded or architecture mismatches.
    """
    print(f"Loading 1-Channel ECGFounder from {weights_path}...")

    # Initialize model with in_channels=1
    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=150,
        use_bn=True,
        use_do=True,
        return_features=True,
    ).to(device)

    try:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Clean module prefix
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(clean_state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load weights: {e}")
        sys.exit(1)


# Initialize global model
model = load_1ch_model(WEIGHTS_PATH)


def preprocess_batch(batch_np, target_length=256):
    """
    Preprocesses a batch of raw time-series signals for the foundation model.

    The preprocessing steps are:
    1. Convert Numpy array to PyTorch Tensor.
    2. Add channel dimension (Batch, Length) -> (Batch, 1, Length).
    3. Z-Score Normalize (Standardize) per sample.
    4. Linearly interpolate (stretch/shrink) the signal to the target_length.

    Args:
        batch_np (np.ndarray): Input batch of shape (Batch_Size, Sequence_Length).
        target_length (int, optional): The length to resize signals to.
                                       Defaults to 1024.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (Batch, 1, Target_Length)
                      moved to the configured device.
    """
    # Convert to Tensor
    # Shape: (Batch, Original_Length)
    x = torch.tensor(batch_np, dtype=torch.float32)

    # Add Channel & Length dimensions for Interpolate
    # Shape needs to be (Batch, Channels, Length)
    x = x.unsqueeze(1)

    # Z-Score Normalize (Required by model)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-6
    x = (x - mean) / std

    # Interpolate (Stretch)
    # This resamples 80 points -> target_length points smoothly
    x = torch.nn.functional.interpolate(
        x, size=target_length, mode="linear", align_corners=False
    )

    return x.to(device)


def make_data_embeddings(data, batch_size=8):
    """
    Runs the embedding extraction pipeline for a specific dataset.

    This function:
    1. Iterates through 'train' and 'test' splits.
    2. Loads the input Parquet file.
    3. Determines the optimal target length:
       - Uses max(256, native_length) to ensure short signals are
         stretched to fit the model's receptive field, while long signals preserve their resolution.
    4. Batches the data, runs inference, and extracts feature embeddings.
    5. Saves the results to a compressed Parquet file.

    Args:
        data (str): The name of the dataset directory.
        batch_size (int, optional): Number of samples per inference batch.
                                    Defaults to 8 (optimized for CPU/Mac).
    """
    for split in ["train", "test"]:
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/ECGFounder-{data}-{split}.parquet"

        if not os.path.exists(INPUT_PATH):
            print(f"Skipping {INPUT_PATH} (Not found)")
            continue

        print(f"\nProcessing {data}-{split} on {device}...")

        X_df = pd.read_parquet(INPUT_PATH)
        X_np = X_df.values

        n = X_df.shape[0]
        m = X_df.shape[1]

        # The model wants the data to be at least 256 observations long
        TARGET_LEN = max(256, m)

        print(f"Input Length: {m} | Target Length: {TARGET_LEN}")

        all_embeddings = []

        for i in tqdm(range(0, n, batch_size), desc="Extracting"):
            batch = X_np[i : i + batch_size]

            inputs = preprocess_batch(batch, target_length=TARGET_LEN)

            with torch.no_grad():
                _, features = model(inputs)

            all_embeddings.append(features.cpu())

        final_tensor = torch.cat(all_embeddings, dim=0)
        final_array = final_tensor.numpy()

        cols = [f"emb_{i}" for i in range(final_array.shape[1])]
        emb_df = pd.DataFrame(final_array, columns=cols, index=X_df.index)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
        print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    make_data_embeddings("SP-500", 8)
    make_data_embeddings("ptb-xl", 8)
    make_data_embeddings("esc-50", 8)
