# Purpose:
# Makes ECGFounder embeddings for StarEmbed data.

# Pre-requisites:
# - Requires running process-star.py.
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

# Get model weights
WEIGHTS_PATH = "weights/1_lead_ECGFounder.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading ECGFounder model on {DEVICE}...")


def load_model(weights_path):
    """
    Loads the 1-channel ECGFounder model from a specified weights path.

    Args:
        weights_path (str): The path to the model's .pth weights file.

    Returns:
        torch.nn.Module: The loaded ECGFounder model in evaluation mode.

    Raises:
        SystemExit: If the weights file fails to load or if a size mismatch
                    occurs during state dict loading.
    """
    # Initialize 1-Channel Model
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
        return_features=True,  # Key: Returns embeddings
    ).to(DEVICE)

    try:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load weights: {e}")
        sys.exit(1)


# Load Model Globally
model = load_model(WEIGHTS_PATH)


def normalize(x, eps=1e-5):
    """
    Normalizes tensor rowwise using Z-score normalization.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        eps (float, optional): A small epsilon value to prevent division by zero.
                              Defaults to 1e-5.

    Returns:
        torch.Tensor: The normalized tensor with mean 0 and standard deviation 1.
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + eps
    return (x - mean) / std


def make_data_embeddings(data):
    """
    Generates and saves ECGFounder embeddings for specified datasets.

    This function processes 'test' and 'train' splits of the given data,
    applies preprocessing (z-score normalization, interpolation) to each row,
    and extracts features using the loaded ECGFounder model. The resulting
    embeddings are saved as Parquet files.

    Args:
        data (str): The name of the dataset (e.g., "star").
    """
    for split in ["test", "train"]:
        print(f"\nProcessing {data}-{split} on {DEVICE}...")

        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/ECGFounder-{data}-{split}.parquet"

        try:
            X_df = pd.read_parquet(INPUT_PATH)
        except FileNotFoundError:
            print(f"Skipping {INPUT_PATH} (File not found)")
            continue

        make_split_embeddings(X_df, OUTPUT_PATH)


def make_split_embeddings(X_df, output_path):
    """
    Iterates through rows of a DataFrame and generates embeddings.

    This function processes each row in the DataFrame's 'target' column,
    generates embeddings using the ECGFounder model, and applies STAR-specific
    pair logic by concatenating adjacent rows (Band A + Band B) before saving.

    Args:
        X_df (pd.DataFrame): The input DataFrame containing the 'target' column
                            with time-series data.
        output_path (str): The path where the output Parquet file will be saved.
    """

    # 'target' column usually contains numpy arrays or lists
    X_arr = X_df["target"]

    all_pooled_embeddings = []

    # Use Inference Mode for speed
    with torch.inference_mode():
        for i in tqdm(range(0, X_arr.shape[0]), desc="Extracting"):
            # Get single row data
            row_data = X_arr.iloc[i] if hasattr(X_arr, "iloc") else X_arr[i]

            make_row_embeddings(
                row_data,
                all_pooled_embeddings,
            )

    # Concatenate and Save
    print("Saving...")
    final_tensor = torch.cat(all_pooled_embeddings, dim=0)
    concatenated_tensor = final_tensor.reshape(-1, 2 * final_tensor.shape[-1])

    # Create a new index, taking the index of the first row in each pair
    new_index = X_df.index[::2][: concatenated_tensor.shape[0]]

    final_array = concatenated_tensor.cpu().numpy()

    print(f"Creating DataFrame with shape {final_array.shape}...")
    cols = [f"emb_{i}" for i in range(final_array.shape[1])]
    emb_df = pd.DataFrame(final_array, columns=cols, index=new_index)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    emb_df.to_parquet(output_path, compression="zstd")
    print(f"Successfully saved to {output_path}")


def make_row_embeddings(row_data, all_pooled_embeddings):
    """
    Adapts a single star light curve for ECGFounder and extracts embeddings.

    This function processes a single row of time-series data by:
    1. Converting to tensor
    2. Adding batch and channel dimensions
    3. Applying Z-score normalization
    4. Interpolating short sequences to minimum length (256)
    5. Extracting features using the ECGFounder model

    Args:
        row_data (np.ndarray, list, or torch.Tensor): The input time-series data
                                                       for a single observation.
        all_pooled_embeddings (list): A list to append the extracted embeddings to.
    """
    # Convert to Tensor
    # Ensure float32 for model compatibility
    tensor_raw = torch.tensor(row_data, dtype=torch.float32).to(DEVICE)

    # Handle Dimensions: (Length) -> (1, 1, Length)
    # Batch=1, Channel=1
    tensor_in = tensor_raw.unsqueeze(0).unsqueeze(0)

    # Normalize (Z-Score)
    # Crucial for CNNs trained on normalized ECGs
    tensor_norm = normalize(tensor_in)

    # Adaptive Interpolation
    # If the light curve is very short (<256), the CNN will downsample it to nothing.
    # We stretch it to 256. If it's long, we keep it as is.
    curr_len = tensor_norm.shape[-1]
    MIN_LEN = 256

    if curr_len < MIN_LEN:
        tensor_final = torch.nn.functional.interpolate(
            tensor_norm, size=MIN_LEN, mode="linear", align_corners=False
        )
    else:
        tensor_final = tensor_norm

    # Forward Pass
    # Model returns (logits, embeddings) -> We take index [1]
    _, features = model(tensor_final)

    # Move to CPU immediately
    all_pooled_embeddings.append(features.cpu())


if __name__ == "__main__":
    make_data_embeddings("star")
