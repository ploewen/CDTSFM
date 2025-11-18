# Purpose:
# Transforms raw data into the appropriate training and testing data while also
# creating the author recomended splits for cross validation.

# Code accessed from:
# https://physionet.org/content/ptb-xl/1.0.3/example_physionet.py

# Pre-requisites:
# - Requires ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip to
#   have been downloaded and extracted to
#   data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/

# Authors:
# - Code originally authored by Wagner et al. (2020).
# - Modified by Philip Loewen to save training and testing data.

# References:
# Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T.
# (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3).
# PhysioNet. RRID:SCR_007345.

# Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W.,
# Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data.
# https://doi.org/10.1038/s41597-020-0495-6

# Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... &
# Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
# Components of a new research resource for complex physiologic signals.
# Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

import pandas as pd
import numpy as np
import wfdb
import ast
import os
import sys

print("Processing files...")


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# Path modified to fit project
path = "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
sampling_rate = 100

# Cause script to end if required files are not found
if not os.path.exists(path):
    print(f"\033[31m{'PTB-XL files not found.'}\033[0m")
    sys.exit(1)

# load and convert annotation data
Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# Flatten the 3D arrays into 2D arrays before converting to DataFrames
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten last two dimensions
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten last two dimensions


# Data tests to check files are the right shape
if not (X_train_flat.shape[1] + X_test_flat.shape[1] == 12 * 2000):
    print(f"\033[31m{'X data has the wrong number of columns.'}\033[0m")
    sys.exit(1)

if not (X_train_flat.shape[0] + X_test_flat.shape[0] == 21799):
    print(f"\033[31m{'X data has the wrong number of rows.'}\033[0m")
    sys.exit(1)

if not (y_train.shape[0] + y_test.shape[0] == 21799):
    print(f"\033[31m{'y data has the wrong number of rows.'}\033[0m")
    sys.exit(1)

print("Done processing files.")

# Make the output directory if it does not yet exists
output_path = "data/processed/ptb-xl"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Convert flattened NumPy arrays and lists to Pandas DataFrames
X_train_df = pd.DataFrame(X_train_flat)
X_test_df = pd.DataFrame(X_test_flat)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

print("Saving files to parquet...")

# Save DataFrames to Parquet files
X_train_df.to_parquet(os.path.join(output_path, "pcb-xl-X-train.parquet"))
print("Saved X_train to pcb-xl-X-train.parquet.")

X_test_df.to_parquet(os.path.join(output_path, "pcb-xl-X-test.parquet"))
print("Saved X_test to pcb-xl-X-test.parquet.")

y_train_df.to_parquet(os.path.join(output_path, "pcb-xl-y-train.parquet"))
print("Saved y_train to pcb-xl-y-train.parquet.")

y_test_df.to_parquet(os.path.join(output_path, "pcb-xl-y-test.parquet"))
print("Saved y_test to pcb-xl-y-test.parquet.")

print("Done saving files.")
