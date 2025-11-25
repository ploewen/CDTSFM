# Purpose:
# Pads StarEmbed dataset.
# Stacks green and red lightwave data.
# Saves datasets.

# Authors:
# - Code written by Philip Loewen

# Pre-requisites:
# - Requires running download-star.py

import pandas as pd
import numpy as np
from tqdm import tqdm


def pad_series(data):
    copy = data.copy()
    for c in ["g", "r"]:
        times = copy[f"{c}_mjd"]
        obs = copy[f"{c}_target"]

        if len(times) == 0:
            copy[f"{c}_mjd"] = np.array([])
            copy[f"{c}_target"] = np.array([])
            continue

        # Normalize time to start from 0 and bin to the nearest lower 0.5-day
        min_time = times[0]
        binned_times = np.floor((times - min_time) * 2) / 2

        if len(binned_times) == 0:
            max_binned_time = 0
        else:
            max_binned_time = binned_times[-1]

        # Create a new, evenly spaced time grid
        new_times_grid = np.arange(0, max_binned_time + 0.5, 0.5)

        # Group observations by their binned time and calculate the mean
        temp_df = pd.DataFrame({"binned": binned_times, "obs": obs})
        binned_means = temp_df.groupby("binned")["obs"].mean()

        # Map the calculated means onto the new time grid, filling empty bins with 0
        new_values_series = binned_means.reindex(new_times_grid, fill_value=0)

        # Replace the old data with the new resampled data
        copy[f"{c}_mjd"] = new_values_series.index.to_list()
        copy[f"{c}_target"] = new_values_series.values.tolist()

    return copy


for split in ["train", "test"]:
    df = pd.read_parquet(f"data/raw/star/star-X-{split}.parquet")
    print(f"Processing {split}ing data...")
    n = df.shape[0]
    new_data = {}
    processed_rows = [
        pad_series(df.iloc[i])
        for i in tqdm(range(n), desc=f"Processing {split} rows...")
    ]

    final_data = pd.DataFrame(processed_rows)

    # Select the 'g_target' and 'r_target' columns
    g_targets = final_data["g_target"]
    r_targets = final_data["r_target"]

    # Concatenate the two Series to stack them vertically
    all_targets = pd.concat([g_targets, r_targets], ignore_index=True)

    # Convert the resulting Series to a DataFrame with a single 'target' column
    stacked_data = all_targets.to_frame(name="target")

    output_path = f"data/processed/star/star-X-{split}.parquet"

    stacked_data.to_parquet(output_path, compression="zstd")
    print(f"Saved to {output_path}.")

print("Processing complete.")
