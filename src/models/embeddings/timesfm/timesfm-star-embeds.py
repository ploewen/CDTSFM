# Purpose:
# Makes TimesFM embeddings for StarEmbed Dataset.

# Pre-requisites:
# - Requires running process-star.py

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


def make_data_embeddings(data):
    """Makes TimesFM embeddings for dataset.

    Args:
        data (str): Name of dataset to embed.
        sequence_length (int): Length of time series.
    """

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
        make_split_embeddings(X_df, OUTPUT_PATH)


def make_split_embeddings(X_df, output_path):
    """Makes TimesFM embeddings for split.

    Args:
        X_df (DataFrame): DataFrame to embed.
        output_path (str): Location to saved embedding.
    """

    # Convert to Tensor
    X_arr = X_df["target"]

    # Pre-allocate output list
    all_pooled_embeddings = []

    # Use Inference Mode (Faster than no_grad)
    with torch.inference_mode():
        for i in tqdm(range(0, X_arr.shape[0]), desc="Extracting"):
            make_row_embeddings(
                X_arr,
                i,
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
    # Create Column Names
    cols = [f"emb_{i}" for i in range(final_array.shape[1])]
    emb_df = pd.DataFrame(final_array, columns=cols, index=new_index)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    emb_df.to_parquet(output_path, compression="zstd")
    print(f"Successfully saved to {output_path}")


def make_row_embeddings(X_arr, i, all_pooled_embeddings):
    """Makes TimesFM embeddings for row.

    Args:
        X_arr (Tensor): Tensor to embed.
        i (int): row iterate.
        all_embeddings (list): List containing all
    """
    # Prepare the batch
    batch_raw = torch.Tensor(X_arr[i]).to(device, non_blocking=True)

    current_len = len(batch_raw)
    if current_len % PATCH_LEN != 0:
        pad_len = PATCH_LEN - (current_len % PATCH_LEN)
        sequence_length = current_len + pad_len
    else:
        sequence_length = current_len

    num_patches = sequence_length // PATCH_LEN

    # Handle padding if the actual data column width < expected sequence_length
    if batch_raw.shape[0] < sequence_length:
        padding = torch.zeros(
            sequence_length - batch_raw.shape[0],
            device=device,
        )
        batch_raw = torch.cat([batch_raw, padding], dim=0)

    # Normalize the data
    batch_norm = normalize(batch_raw)

    # Reshape the batch
    # Input: (Batch, Time) -> Output: (Batch, Num_Patches, Patch_Len)
    batch_patched = batch_norm.reshape(1, num_patches, PATCH_LEN)

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
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    stdev = torch.sqrt(var + eps)
    x_norm = (x - mean) / stdev
    return x_norm


if __name__ == "__main__":
    make_data_embeddings("star")
