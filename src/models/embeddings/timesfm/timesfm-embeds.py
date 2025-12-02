# Purpose:
# Makes TimesFM embeddings for S&P 500, ESC 50, PTB XL Datasets.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py

# Authors:
# - Code written by Philip Loewen

# Reference:
# A. Das, W. Kong, R. Sen, Y. Zhou, (2024), A decoder-only foundation model for
# time-series forecasting, Proceedings of the 41st International Conference on
# Machine Learning

import torch
import timesfm
import pandas as pd
from tqdm import tqdm
import os

# Set device to gpu if possible
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load TimesFM 2.5
wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", torch_compile=True
)
model = wrapper.model
model.to(device)
model.eval()

# Constants from the model config
PATCH_LEN = 32  # Standard for TimesFM


def make_data_embeddings(data, batch_size, sequence_length):
    """Makes TimesFM embeddings for dataset.

    Args:
        data (str): Name of dataset to embed.
        batch_size (int): Number of rows to pass through model at once.
        sequence_length (int): Length of time series.
    """
    # Ensure sequence length is divisible by patch length
    if sequence_length % PATCH_LEN != 0:
        pad_len = PATCH_LEN - (sequence_length % PATCH_LEN)
        sequence_length += pad_len
        print(f"Adjusted sequence_length to {sequence_length} to fit patch size.")

    # Number of patches per sequence
    num_patches = sequence_length // PATCH_LEN

    for split in ["test", "train"]:
        print(f"\nProcessing {data}-{split} on {device}...")

        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/TIMESFM-{data}-{split}.parquet"

        # If file isn't found skip it
        try:
            X_df = pd.read_parquet(INPUT_PATH)
        except FileNotFoundError:
            print(f"Skipping {INPUT_PATH} (File not found)")
            continue
        make_split_embeddings(
            X_df, batch_size, sequence_length, num_patches, OUTPUT_PATH
        )


def make_split_embeddings(X_df, batch_size, sequence_length, num_patches, output_path):
    """Makes TimesFM embeddings for split.

    Args:
        X_df (DataFrame): DataFrame to embed.
        batch_size (int): Number of rows to pass through model at once.
        sequence_length (int): Length of time series.
        num_patches (int): Number of patches for given time series (sequence_length // PATCH_LEN).
        output_path (str): Location to saved embedding.
    """

    # Convert to Tensor
    X_arr = torch.from_numpy(X_df.values).float()

    # Pre-allocate output list
    all_pooled_embeddings = []

    # Use Inference Mode (Faster than no_grad)
    with torch.inference_mode():
        for i in tqdm(range(0, len(X_arr), batch_size), desc="Extracting"):
            make_batch_embeddings(
                X_arr,
                i,
                batch_size,
                sequence_length,
                num_patches,
                all_pooled_embeddings,
            )

    # Concatenate and Save
    print("Saving...")
    final_tensor = torch.cat(all_pooled_embeddings, dim=0)
    final_array = final_tensor.numpy()

    cols = [f"emb_{i}" for i in range(final_array.shape[1])]
    emb_df = pd.DataFrame(final_array, columns=cols, index=X_df.index)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    emb_df.to_parquet(output_path, compression="zstd")
    print(f"Saved to {output_path}")


def make_batch_embeddings(
    X_arr, i, batch_size, sequence_length, num_patches, all_pooled_embeddings
):
    """Makes TimesFM embeddings for batch.

    Args:
        X_arr (Tensor): Tensor to embed.
        i (int): Batch iterate.
        batch_size (int): Number of rows to pass through model at once.
        sequence_length (int): Length of time series.
        num_patches (int): Number of patches for given time series (sequence_length // PATCH_LEN).
        all_embeddings (list): List containing all
    """
    # Prepare the batch
    batch_raw = X_arr[i : i + batch_size].to(device, non_blocking=True)

    # Handle padding if the actual data column width < expected sequence_length
    if batch_raw.shape[1] < sequence_length:
        padding = torch.zeros(
            batch_raw.shape[0],
            sequence_length - batch_raw.shape[1],
            device=device,
        )
        batch_raw = torch.cat([batch_raw, padding], dim=1)

    # Normalize the data
    batch_norm = normalize(batch_raw)

    # Reshape the batch
    # Input: (Batch, Time) -> Output: (Batch, Num_Patches, Patch_Len)
    batch_patched = batch_norm.reshape(batch_raw.shape[0], num_patches, PATCH_LEN)

    # Create Masks
    # 1 = Observed. Assuming all data present.
    batch_mask = torch.ones_like(batch_patched, dtype=torch.float32)

    # Do Forward Pass for batch
    # Returns tuple: (input_emb, output_emb, forecast_output, ...)
    outputs, _ = model(batch_patched, batch_mask)

    # The second element is the hidden state (Batch, Patches, Dim)
    hidden_states = outputs[1]

    # Collapse patches into one vector by average embeddings
    pooled = hidden_states.mean(dim=1)

    # Move back to CPU immediately to free GPU
    all_pooled_embeddings.append(pooled.cpu())


def normalize(x, eps=1e-5):
    """Normalizes tensor rowwise.

    Args:
        x (Tensor): Tensor to normalize.
        eps (float, optional): Stabilizing constant. Defaults to 1e-5.

    Returns:
        Tensor_: normalized tensor.
    """
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    stdev = torch.sqrt(var + eps)
    x_norm = (x - mean) / stdev
    return x_norm


if __name__ == "__main__":
    make_data_embeddings("SP-500", batch_size=64, sequence_length=80)
    make_data_embeddings("ESC-50", batch_size=1, sequence_length=80000)
    make_data_embeddings("ptb-xl", batch_size=16, sequence_length=12000)
