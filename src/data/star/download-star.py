# Purpose:
# Retrieves S&P 500 stock tickers from Wikipedia.
# Downloads stock data from January 1st 2000 to November 1st 2025 for all retrieved tickers
# Computes daily simple return of stocks
# Saves return files to parquet

# Authors:
# - Code written by Philip Loewen

# Pre-requisites:
# - Requires setting up hugginface-cli

# Reference:
# Li, W., Chen, H., Lin, Q., Rehemtulla, N., Shah, V. G., Wu, D., Miller, A. A., & Liu, H. (2025).
# StarEmbed: Benchmarking Time Series Foundation models on astronomical observations of
# variable stars. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2510.06200

from datasets import load_dataset
import pandas as pd
import os

# Download huggingface dataset
print("Downloading data...")
ds = load_dataset("123anonymous123/StarEmbed")
print("Done downloading data.")

# If raw data folder does not exist, create it
output_path = "data/raw/star"
if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Saving data...")
train_data = pd.DataFrame(ds["train"])
val_data = pd.DataFrame(ds["validation"])
merged_data = pd.concat([train_data, val_data])

merged_data.to_parquet(os.path.join(output_path, "train-star-raw.parquet"))
print("Saved training data.")
ds["test"].to_parquet(os.path.join(output_path, "test-star-raw.parquet"))
print("Saved testing data.")

print("Done saving data.")
