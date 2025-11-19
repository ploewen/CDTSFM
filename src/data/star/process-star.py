# Purpose:
# Makes training and testing dataframes, and splits input and output variables for star data.

# Pre-requisites:
# - Requires running download-star.py

# Authors:
# - Code written by Philip Loewen

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
    bands = data["bands_data"].reset_index(drop=True)
    for i in range(n):
        d = dict[i] = {}
        i_data = bands[i]
        for c in ("g", "r"):
            add_observation_to_dict(d, c, i_data)


def main():
    input_folder = "data/raw/star/"
    output_folder = "data/processed/star/"

    # If output folder does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    splits = ["train", "test"]
    for split in splits:
        print(f"Processing {split} data...")
        # Read split data
        input_file = input_folder + split + "-star-raw.parquet"
        data = pd.read_parquet(input_file)

        # Build the dictionary for input data
        dict = build_dict_for_split(data)

        # Transform to dataframes
        X_data = pd.DataFrame(dict).transpose()
        y_data = pd.DataFrame(data["class_str"])

        # Split file paths
        X_output = output_folder + "star-X-" + split + ".parquet"
        y_output = output_folder + "star-y-" + split + ".parquet"

        # Save files.
        X_data.to_parquet(X_output)
        print(f"Saved X_{split} to {X_output}.")
        y_data.to_parquet(y_output)
        print(f"Saved y_{split} to {y_output}.")

    print("Done processing and saving data.")


if __name__ == "__main__":
    main()
