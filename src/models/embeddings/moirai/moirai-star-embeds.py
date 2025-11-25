# Purpose:
# Makes Moirai embeddings for StarEmbed data.

# Pre-requisites:
# - Requires running process-star.py

# Authors:
# - Code written by Philip Loewen

# Reference:
# Woo et al., (2024), Unified Training of Universal Time Series Forecasting Transformers,
# arXiv preprint arXiv:2402.02592

import pandas as pd
import torch
import numpy as np
from uni2ts.model.moirai2 import Moirai2Module
from tqdm import tqdm
import os

# Set the device to use GPU of possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model to moirai 2 small
model = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
model.to(device)
model.eval()


def make_data_embeddings(data):
    PATCH_SIZE = 16  # Fixed by Model

    for split in ["train", "test"]:
        print(f"\nProcessing {data}-{split} on {device}...")
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/MOIRAI-{data}-{split}.parquet"

        # If file isn't found skip it
        try:
            X_df = pd.read_parquet(INPUT_PATH)
        except FileNotFoundError:
            print(f"Skipping {INPUT_PATH} (File not found)")
            continue

        all_pooled_embeddings = make_split_embeddings(PATCH_SIZE, X_df)

        # Save the results
        print("Concatenating results...")
        save_embeddings(OUTPUT_PATH, X_df, all_pooled_embeddings)


def make_split_embeddings(patch_size, X_df):
    X_arr = X_df["target"]
    # Hook storage
    embeddings_storage = {}

    model.encoder.register_forward_hook(
        get_embeddings(embeddings_storage, "backbone_output")
    )

    # Get all the embeddings into a list
    all_pooled_embeddings = []

    print("Making embeddings...")
    # Loop through data by row
    for i in tqdm(range(0, X_arr.shape[0])):
        make_row_embeddings(
            X_arr,
            i,
            patch_size,
            embeddings_storage,
            all_pooled_embeddings,
        )

    return all_pooled_embeddings


def make_row_embeddings(
    X_arr,
    i,
    patch_size,
    embeddings_storage,
    all_pooled_embeddings,
):
    inputs = prepare_inputs(X_arr, i, patch_size)

    # Pass our inputs into the model
    with torch.no_grad():
        model(**inputs)

    # Extract embeddings
    raw_emb = embeddings_storage["backbone_output"]

    # Go From [Batch, 5, 384] -> [Batch, 384]
    pooled_emb = raw_emb.mean(dim=1)

    # Add pooled embedding to list
    all_pooled_embeddings.append(pooled_emb)


def prepare_inputs(X_arr, i, patch_size):
    # Adjust SEQ_LEN to be divisible by PATCH_SIZE for the model
    batch_raw = X_arr[i]
    current_len = len(batch_raw)
    if current_len % patch_size != 0:
        pad_len = patch_size - (current_len % patch_size)
        seq_len = current_len + pad_len
    else:
        seq_len = current_len

    num_patches = seq_len // patch_size
    # Prepare batch
    current_bs = 1

    padded_batch = get_padded(current_len, seq_len, batch_raw)

    # Reshape the batch: Input: (Batch, 80) -> Output: (Batch, 5, 16)
    target_patched = torch.tensor(padded_batch, dtype=torch.float32).reshape(
        current_bs, num_patches, patch_size
    )

    # Create Masks & IDs
    true_len = min(current_len, seq_len)
    mask = np.arange(seq_len) < true_len
    observed_mask = torch.from_numpy(
        np.broadcast_to(mask, (current_bs, seq_len)).copy()
    ).reshape(current_bs, num_patches, patch_size)

    # Time ID: [0, 1, 2, 3, 4] repeated for batch
    time_id = torch.arange(num_patches, dtype=torch.int64).expand(current_bs, -1)

    # Zeros for others
    variate_id = torch.zeros((current_bs, num_patches), dtype=torch.int64)
    sample_id = torch.zeros((current_bs, num_patches), dtype=torch.int64)
    prediction_mask = torch.zeros((current_bs, num_patches), dtype=torch.bool)

    # Move to Device
    inputs = {
        "target": target_patched.to(device),
        "observed_mask": observed_mask.to(device),
        "sample_id": sample_id.to(device),
        "time_id": time_id.to(device),
        "variate_id": variate_id.to(device),
        "prediction_mask": prediction_mask.to(device),
        "training_mode": False,
    }
    return inputs


def get_padded(current_len, seq_len, batch_raw):
    # Pad or truncate the batch to SEQ_LEN
    if current_len < seq_len:
        pad_width = seq_len - current_len
        padded_batch = np.pad(
            batch_raw, ((0, pad_width)), "constant", constant_values=0
        )
    elif current_len > seq_len:
        padded_batch = batch_raw[:seq_len]
    else:
        padded_batch = batch_raw
    return padded_batch


def save_embeddings(output_path, X_df, all_pooled_embeddings):
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


def get_embeddings(embeddings_storage, name):
    def hook(model, input, output):
        embeddings_storage[name] = output.detach().cpu()

    return hook


if __name__ == "__main__":
    make_data_embeddings("star")
