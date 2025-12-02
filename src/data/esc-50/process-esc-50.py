# Purpose:
# Transforms raw data into the appropriate training and testing data while also
# creating the author recomended splits for cross validation.

# Pre-requisites:
# - Requires ESC-50-master.zip to have been downloaded and extracted to ESC-50-master/

# Authors:
# - Code written by Philip Loewen

# References:
# K. J. Piczak. ESC: Dataset for Environmental Sound Classification.
# Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.

import librosa
import pandas as pd
import numpy as np
import os
import sys

path = "data/raw/ESC-50-master/audio/"

# Cause script to end if required files are not found
if not os.path.exists(path):
    print(f"\033[31m{'ESC-50 files not found.'}\033[0m")
    sys.exit(1)

print("Processing files...")


sr = 16000  # Resample to 8 kHz
expected_length = sr * 5
data = []
folds = []
targets = []

# For each file get its signal data, split and target
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    signal, _ = librosa.load(file_path, sr=sr)
    data.append(signal)

    # The file name is formatted is {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
    fold, _, _, target = file.split("-")
    folds.append(int(fold))
    targets.append(int(target[:-4]))

# Make the appropriate dataframes
X = pd.DataFrame(np.vstack(data))
Y = pd.DataFrame(targets)
folds = np.array(folds)

# Choose fold 5 to be the test fold
test_fold = 5
X_train = X[folds != test_fold]
y_train = Y[folds != test_fold]
X_test = X[folds == test_fold]
y_test = Y[folds == test_fold]

# Data tests to check files are the right shape
if not (X_train.shape == (1600, expected_length)):
    print(
        f"\033[31m{'X train data has the wrong shape. '}\033[0m\033[31m{X_train.shape}\033[0m"
    )
    print(f"\033[31m{'Shape should be  be (1600, '}{expected_length}{')'}\033[0m")
    sys.exit(1)

if not (X_test.shape == (400, expected_length)):
    print(
        f"\033[31m{'X test data has the wrong shape. '}\033[0m\033[31m{X_test.shape}\033[0m"
    )
    print(f"\033[31m{'Shape should be  be (400, '}{expected_length}{')'}\033[0m")
    sys.exit(1)

if not (y_train.shape == (1600, 1)):
    print(
        f"\033[31m{'y train data has the wrong shape. '}\033[0m\033[31m{y_train.shape}\033[0m"
    )
    print(f"\033[31m{'Shape should be  be (1600, 1)'}\033[0m")
    sys.exit(1)

if not (y_test.shape == (400, 1)):
    print(
        f"\033[31m{'y test data has the wrong shape. '}\033[0m\033[31m{y_test.shape}\033[0m"
    )
    print(f"\033[31m{'Shape should be  be (400, 1)'}\033[0m")
    sys.exit(1)

print("Done processing files.")

print("Saving files to parquet...")

# Make the output directory if it does not yet exists
output_path = "data/processed/esc-50"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save DataFrames to Parquet files
X_train.to_parquet(os.path.join(output_path, "esc-50-X-train.parquet"))
print("Saved X_train to esc-50-X-train.parquet.")

X_test.to_parquet(os.path.join(output_path, "esc-50-X-test.parquet"))
print("Saved X_test to esc-50-X-test.parquet.")

y_train.to_parquet(os.path.join(output_path, "esc-50-y-train.parquet"))
print("Saved y_train to esc-50-y-train.parquet.")

y_test.to_parquet(os.path.join(output_path, "esc-50-y-test.parquet"))
print("Saved y_test to esc-50-y-test.parquet.")

print("Done saving files.")
