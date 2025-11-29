# Purpose:
# Makes ASTROMER embeddings for S&P 500, ESC 50, PTB XL Datasets.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py

# Authors:
# - Code written by Philip Loewen

# Reference:
# C. Donoso-Oliva, I. Becker, P. Protopapas, G. Cabrera-Vives, M. Vishnu, and H. Vardhan,
# “ASTROMER,” Astronomy and Astrophysics, vol. 670, p. A54, Dec. 2022,
# doi: 10.1051/0004-6361/202243928.

import os

# FORCE LEGACY KERAS (Required for ASTROMER)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import math
import time
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from ASTROMER.models import SingleBandEncoder

# Check for GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU Detected: {gpus[0]}")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    print("\033[93mWARNING: No GPU detected. Running on CPU.\033[0m")


def make_data_embeddings(data, model, batch_size=2):
    for split in ["train", "test"]:
        print(f"\nProcessing {data}-{split}...")
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/ASTRO-{data}-{split}.parquet"

        if not os.path.exists(INPUT_PATH):
            print(f"Skipping {INPUT_PATH} (File not found)")
            continue

        X_df = pd.read_parquet(INPUT_PATH)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

        print("  Vectorizing and reshaping data...")
        # Create Dictionary of giant arrays
        augm_data_dict, lengths_full = reshape_data(X_df)
        print(f"  Reshaping complete. Input keys: {list(augm_data_dict.keys())}")

        cols = None
        n = len(X_df)
        num_batches = math.ceil(n / batch_size)

        writer = None
        total_time = 0
        current_segment_idx = 0

        # Access the underlying Keras model directly for speed
        layer_name = "encoder"  # The internal name ASTROMER gives its transformer block
        keras_model = tf.keras.Model(
            inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
        )

        try:
            for i in range(0, n, batch_size):
                t0_batch = time.time()
                current_batch_num = (i // batch_size) + 1

                current_elapsed = total_time + (time.time() - t0_batch)
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(current_elapsed))

                if current_batch_num > 1:
                    avg_time = total_time / (current_batch_num - 1)
                    remaining = num_batches - (current_batch_num - 1)
                    eta_str = time.strftime(
                        "%H:%M:%S", time.gmtime(avg_time * remaining)
                    )
                else:
                    eta_str = "--:--:--"

                if current_batch_num % 20 == 0 or current_batch_num == 1:
                    print(
                        f"\r  Processing Batch {current_batch_num}/{num_batches} | Elapsed: {elapsed_str} | ETA: {eta_str}      ",
                        end="",
                    )

                # Get batch lengths
                batch_lengths = lengths_full[i : i + batch_size]
                num_segments_in_batch = sum(batch_lengths)

                # Slice Dictionary Keys
                start = current_segment_idx
                end = current_segment_idx + num_segments_in_batch

                # Create the batch dictionary by slicing every array in the main dict
                batch_dict = {
                    key: val[start:end] for key, val in augm_data_dict.items()
                }
                current_segment_idx += num_segments_in_batch

                # Pass batch through the model to generate embeddings with shape (batch_size*num_segments, 200, 256)
                embeds = keras_model.predict_on_batch(batch_dict)

                # AGGREGATING (batch_size*num_segments, 200, 256) -> (batch_size, 256)
                final_array = aggregate_embeddings(batch_lengths, embeds)

                if final_array.size == 0:
                    continue

                if cols is None:
                    cols = [f"emb_{j}" for j in range(final_array.shape[1])]

                batch_df_index = X_df.index[i : i + batch_size]
                emb_df = pd.DataFrame(final_array, columns=cols, index=batch_df_index)

                # Save batch to file
                table = pa.Table.from_pandas(emb_df)
                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT_PATH, table.schema, compression="zstd"
                    )
                writer.write_table(table=table)

                total_time += time.time() - t0_batch

        finally:
            if writer:
                writer.close()

        print(f"\nSuccessfully saved to {OUTPUT_PATH}")


def aggregate_embeddings(lengths, embeds):
    """
    Optimized aggregation using Pure NumPy.
    """
    if hasattr(embeds, "numpy"):
        embeds = embeds.numpy()

    # Pool over time dimension (dim 1)
    pooled_embeds = np.mean(embeds, axis=1)

    if all(x == 1 for x in lengths):
        return pooled_embeds

    # Pool across windows to get series embeddings
    final_pooled_embeds_list = []
    k = 0
    for length in lengths:
        sample_chunk = pooled_embeds[k : k + length]
        sample_mean = np.mean(sample_chunk, axis=0)
        final_pooled_embeds_list.append(sample_mean)
        k += length

    return np.array(final_pooled_embeds_list)


def reshape_data(X_df):
    """
    Creates the dictionary inputs expected by ASTROMER's underlying Keras model.
    Keys: 'input', 'times', 'mask_in', 'length'
    """
    vals = X_df.values
    n, m = vals.shape
    window = 200

    if m > window:
        n_segments = m // window
        vals = vals[:, : n_segments * window]
        vals = vals.reshape(n, n_segments, window)
        vals_reshaped = vals.reshape(-1, window)
        lengths = [n_segments] * n
    else:
        vals_reshaped = vals
        lengths = [1] * n

    # Input (Magnitude) - Shape (N, 200, 1)
    mag_data = vals_reshaped[..., np.newaxis].astype(np.float32)

    # Times - Shape (N, 200, 1)
    time_steps = np.tile(np.arange(window), (vals_reshaped.shape[0], 1))
    time_data = time_steps[..., np.newaxis].astype(np.float32)

    # Mask - Shape (N, 200, 1) (0 usually means valid/unmasked)
    mask_data = np.zeros_like(mag_data)

    # Length - Shape (N, 1) - The model requires this to know the sequence length
    # Since we are using fixed windows of 200, the length is 200 for everyone.
    length_data = np.full((vals_reshaped.shape[0], 1), window, dtype=np.int32)

    input_dict = {
        "input": mag_data,
        "times": time_data,
        "mask_in": mask_data,
        "length": length_data,
    }

    return input_dict, lengths


if __name__ == "__main__":
    # Initialize Model
    model = SingleBandEncoder()
    model = model.from_pretraining("macho")

    # Run Process
    make_data_embeddings("esc-50", model)
    make_data_embeddings("ptb-xl", model)
    make_data_embeddings("SP-500", model, batch_size=8)
