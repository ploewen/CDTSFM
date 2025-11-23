# Purpose:
# Retrieves StarEmbed dataset.
# Merges training and validation set.
# Retrieves light wave and timestamp data.
# Saves datasets.

# Authors:
# - Code written by Philip Loewen

# Pre-requisites:
# - Requires setting up hugginface-cli

# Reference:
# Li, W., Chen, H., Lin, Q., Rehemtulla, N., Shah, V. G., Wu, D., Miller, A. A., & Liu, H. (2025).
# StarEmbed: Benchmarking Time Series Foundation models on astronomical observations of
# variable stars. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2510.06200

import polars as pl
import pandas as pd
import numpy as np
import os


# Creates a dictionary entry for light colour c and observation i
def add_observation_to_dict(dict, c, i_data):
    c_data = i_data[c]
    target_str = c + "_target"
    mjd_str = c + "_mjd"

    if c_data is None:
        dict[target_str] = np.zeros(300)
        dict[mjd_str] = np.array(range(300))
    else:
        dict[target_str] = c_data["target"]
        dict[mjd_str] = c_data["mjd"]


# Builds a dictionary of input variables for data for one split
def build_dict_for_split(data):
    dict = {}
    n = data.shape[0]
    bands = data["bands_data"]
    for i in range(n):
        d = dict[i] = {}
        i_data = bands[i]
        for c in ("g", "r"):
            add_observation_to_dict(d, c, i_data)

    return dict


# Login using e.g. `huggingface-cli login` to access this dataset
splits = {
    "train": "data/train-*.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# Download huggingface dataset
print("Downloading data...")
train = pl.read_parquet("hf://datasets/123anonymous123/StarEmbed/" + splits["train"])
test = pl.read_parquet("hf://datasets/123anonymous123/StarEmbed/" + splits["test"])
val = pl.read_parquet("hf://datasets/123anonymous123/StarEmbed/" + splits["validation"])

print("Making dataframes...")

# If raw data folder does not exist, create it
output_path = "data/raw/star"
if not os.path.exists(output_path):
    os.makedirs(output_path)

colnames = train.columns

# Merge training and validation sets
merged_train = train.vstack(val)
merged_data = pd.DataFrame(build_dict_for_split(merged_train)).transpose()
test_data = pd.DataFrame(build_dict_for_split(test)).transpose()

# Get star classes
y_merge = pd.DataFrame(merged_train["class_str"], columns=["Class"])
y_test = pd.DataFrame(test["class_str"], columns=["Class"])


print("Saving data...")

# Save training data
merged_data.to_parquet(os.path.join(output_path, "star-X-train.parquet"))
y_merge.to_parquet(os.path.join(output_path, "star-y-train.parquet"))
print("Saved training data.")

# Save testing data
test_data.to_parquet(os.path.join(output_path, "star-X-test.parquet"))
y_test.to_parquet(os.path.join(output_path, "star-y-test.parquet"))
print("Saved testing data.")

print("Done saving data.")
