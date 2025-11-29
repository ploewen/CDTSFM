# Purpose:
# Makes ASTROMER embeddings for StarEmbed data.

# Pre-requisites:
# - Requires running process-star.py

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
        INPUT_PATH = f"data/raw/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/ASTRO-{data}-{split}.parquet"

        if not os.path.exists(INPUT_PATH):
            print(f"Skipping {INPUT_PATH} (File not found)")
            continue

        X_df = pd.read_parquet(INPUT_PATH)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

        cols = None
        n = len(X_df)
        num_batches = math.ceil(n / batch_size)

        writer = None
        total_time = 0

        # Access the underlying Keras model directly for speed
        layer_name = "encoder"  # The internal name ASTROMER gives its transformer block
        keras_model = tf.keras.Model(
            inputs=model.model.input, outputs=model.model.get_layer(layer_name).output
        )

        try:
            for i in range(0, n, batch_size):
                t0_batch = time.time()
                current_batch_num = (i // batch_size) + 1

                current_elapsed = total_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(current_elapsed))

                if current_batch_num > 1:
                    avg_time = total_time / i
                    remaining_samples = n - i
                    eta_seconds = avg_time * remaining_samples
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                else:
                    eta_str = "--:--:--"

                if (
                    current_batch_num % 20 == 0
                    or current_batch_num == 1
                    or current_batch_num == num_batches
                ):
                    print(
                        f"\r  Processing Batch {current_batch_num}/{num_batches} | Elapsed: {elapsed_str} | ETA: {eta_str}      ",
                        end="",
                    )

                batch_df = X_df.iloc[i : i + batch_size]

                # Process first light curve
                dict1, lengths1 = reshape_data(batch_df.iloc[:, 0], batch_df.iloc[:, 1])

                if dict1:
                    embeds1 = keras_model.predict_on_batch(dict1)
                    final_array1 = aggregate_embeddings(lengths1, embeds1)
                else:  # Handle case where whole batch is empty
                    final_array1 = np.zeros((len(batch_df), 256))

                # Process second light curve
                dict2, lengths2 = reshape_data(batch_df.iloc[:, 2], batch_df.iloc[:, 3])

                if dict2:
                    embeds2 = keras_model.predict_on_batch(dict2)
                    final_array2 = aggregate_embeddings(lengths2, embeds2)
                else:
                    final_array2 = np.zeros((len(batch_df), 256))

                # AGGREGATING

                final_array = np.concatenate([final_array1, final_array2], axis=1)

                if final_array.size == 0:
                    continue

                if cols is None:
                    cols = [f"emb_{j}" for j in range(final_array.shape[1])]

                batch_df_index = X_df.index[i : i + batch_size]
                emb_df = pd.DataFrame(final_array, columns=cols, index=batch_df_index)

                # Save to file
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

    final_pooled_embeds_list = []
    k = 0
    for length in lengths:
        sample_chunk = pooled_embeds[k : k + length]
        sample_mean = np.mean(sample_chunk, axis=0)
        final_pooled_embeds_list.append(sample_mean)
        k += length

    return np.array(final_pooled_embeds_list)


def reshape_data(lc_series, time_series):
    """
    Creates the dictionary inputs expected by ASTROMER's underlying Keras model
    for a batch of light curves stored in pandas Series.
    """
    window = 200
    all_mag_segments = []
    all_time_segments = []
    lengths = []

    for lc, ts in zip(lc_series, time_series):
        m = len(lc) if hasattr(lc, "__len__") and lc is not None else 0

        if m == 0:
            # For empty light curves, add one segment of zeros
            all_mag_segments.append(np.zeros(window))
            all_time_segments.append(np.zeros(window))
            lengths.append(1)
            continue

        # Ceiling division to calculate number of segments
        n_segments = (m + window - 1) // window
        padded_len = n_segments * window

        # Pad light curve and time series to fit into segments
        lc_padded = np.zeros(padded_len)
        lc_padded[:m] = lc
        ts_padded = np.zeros(padded_len)
        ts_padded[:m] = ts

        # Reshape into segments and add to list
        all_mag_segments.extend(lc_padded.reshape(n_segments, window))
        all_time_segments.extend(ts_padded.reshape(n_segments, window))
        lengths.append(n_segments)

    if not all_mag_segments:
        return None, []

    # Convert lists to numpy arrays
    vals_reshaped = np.array(all_mag_segments)
    times_reshaped = np.array(all_time_segments)

    # Create model inputs
    mag_data = vals_reshaped[..., np.newaxis].astype(np.float32)
    time_data = times_reshaped[..., np.newaxis].astype(np.float32)
    mask_data = np.zeros_like(mag_data)
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
    make_data_embeddings("star", model, batch_size=8)
