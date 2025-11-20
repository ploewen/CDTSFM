# Purpose:
# Makes training and testing splits, and splits input and output variables for S&P 500 data.

# Pre-requisites:
# - Requires running download-SP-500.py

# Authors:
# - Code written by Philip Loewen

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys


# Split the data into sections of 1000 days
def make_splits(data):
    returns = data.reset_index()["return"]
    n = returns.shape[0]

    splits = []
    if n >= 1000:
        n_splits = n // 1000
        for i in range(n_splits):
            lwr = i * 1000
            upr = (i + 1) * 1000
            split = returns[lwr:upr]
            splits.append(split)
    else:
        raise Exception()
    splits = pd.DataFrame(np.vstack(splits))
    return splits


# From a 1000 day split make a training and testing set
# Training set taken from days 1 - 749
# Testing set taken from days 751 - 1000
# Make windows that are 81 days long.
# First 80 days make the input, 81st days makes the output.
def make_train_test(split):
    windows = []
    for i in range(669):
        window = split[i : 81 + i]
        windows.append(window)
    windows = pd.DataFrame(np.vstack(windows))

    X_train = windows.loc[:, 0:79]
    y_values = windows.loc[:, 80]
    y_train = (y_values > X_train.median(axis=1)).astype(int)

    windows = []
    for i in range(750, 919):
        window = split[i : 81 + i]
        windows.append(window)
    windows = pd.DataFrame(np.vstack(windows))

    X_test = windows.loc[:, 0:79]
    y_values = windows.loc[:, 80]
    y_test = (y_values > X_test.median(axis=1)).astype(int)

    return (X_train, X_test, y_train, y_test)


# Updates the training and testing lists, using func and arg to create new sets
def update_lists(X_train_list, X_test_list, y_train_list, y_test_list, func, arg):
    X_train, X_test, y_train, y_test = func(arg)

    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)


# Makes the dataframes from the training and testing lists
def make_dataframes(X_train_list, X_test_list, y_train_list, y_test_list):
    X_train = pd.DataFrame(np.vstack(X_train_list))
    X_test = pd.DataFrame(np.vstack(X_test_list))
    y_train = pd.DataFrame(np.vstack(y_train_list))
    y_test = pd.DataFrame(np.vstack(y_test_list))

    return (X_train, X_test, y_train, y_test)


# Make the entire training and testing splits for a dataset
def make_splits_from_data(data):
    splits = make_splits(data)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for i in range(splits.shape[0]):
        split = splits.iloc[i]
        update_lists(
            X_train_list, X_test_list, y_train_list, y_test_list, make_train_test, split
        )

    return make_dataframes(X_train_list, X_test_list, y_train_list, y_test_list)


def main():
    path = "data/raw/SP500/"
    # Cause script to end if required files are not found
    if not os.path.exists(path):
        print(f"\033[31m{'S&P 500 files not found.'}\033[0m")
        sys.exit(1)

    print("Processing files...")

    failed = []

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # For each file get its data and create splits
    for file in tqdm(
        os.listdir(path), desc="Making training and testing splits", unit=" files"
    ):
        # For some reason .DS_Store is being read so skip it
        if file == ".DS_Store":
            continue
        try:
            file_path = os.path.join(path, file)
            data = pd.read_parquet(file_path)
            update_lists(
                X_train_list,
                X_test_list,
                y_train_list,
                y_test_list,
                make_splits_from_data,
                data,
            )
        except Exception:
            # If a stock couldn't be processed add it to the list of failed stocks
            ticker = file.split("-")[0]
            failed.append(ticker)
            continue

    # If some stocks couldn't be processed print them to console
    if failed:
        print(
            f"\033[31m{'Warning: Some stocks did not have enough observations:'}\033[0m"
        )
        print(", ".join(failed))

    # Transform the lists to dataframes
    (X_train, X_test, y_train, y_test) = make_dataframes(
        X_train_list, X_test_list, y_train_list, y_test_list
    )

    print("Done processing files.")
    print("Saving files to parquet...")

    # Make the output directory if it does not yet exists
    output_path = "data/processed/SP-500"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save DataFrames to Parquet files
    X_train.to_parquet(os.path.join(output_path, "SP-500-X-train.parquet"))
    print("Saved X_train to SP-500-X-train.parquet.")

    X_test.to_parquet(os.path.join(output_path, "SP-500-X-test.parquet"))
    print("Saved X_test to SP-500-X-test.parquet.")

    y_train.to_parquet(os.path.join(output_path, "SP-500-y-train.parquet"))
    print("Saved y_train to SP-500-y-train.parquet.")

    y_test.to_parquet(os.path.join(output_path, "SP-500-y-test.parquet"))
    print("Saved y_test to SP-500-y-test.parquet.")

    print("Done saving files.")


if __name__ == "__main__":
    main()
