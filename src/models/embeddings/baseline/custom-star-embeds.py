# Purpose:
# Makes custom embeddings for StarEmbed data.

# Pre-requisites:
# - Requires running process-star.py

# Authors:
# - Code written by Philip Loewen
# - Partially adapted from the StarEmbed project

# Reference:
# Li, W., Chen, H., Lin, Q., Rehemtulla, N., Shah, V. G., Wu, D., Miller, A. A., & Liu, H. (2025).
# StarEmbed: Benchmarking Time Series Foundation models on astronomical observations of
# variable stars. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2510.06200
#
# K. L. Malanchev et al., “Anomaly detection in the Zwicky Transient Facility DR3,”
# Monthly Notices of the Royal Astronomical Society, vol. 502, no. 4, pp. 5147–5175,
# Feb. 2021, doi: 10.1093/mnras/stab316.


import light_curve as lc
import logging
import os
import pandas as pd
import numpy as np
import warnings

from cesium import featurize
from tqdm import tqdm
from joblib import Parallel, delayed

# Config
# Downsample threshold: If a light curve has > 500 points,
# we subsample it for the expensive Cesium periodicity check.
CESIUM_MAX_POINTS = 500
BATCH_SIZE = 1000

# Suppress verbose warnings from optimization libraries
warnings.filterwarnings("ignore")
logging.getLogger("cesium").setLevel(logging.ERROR)


# Features to extract using the light-curve package
LC_extractor = lc.Extractor(
    lc.Amplitude(),
    lc.AndersonDarlingNormal(),
    lc.BeyondNStd(nstd=1),
    lc.BeyondNStd(nstd=2),
    lc.BeyondNStd(nstd=3),
    lc.Cusum(),
    lc.Eta(),
    lc.InterPercentileRange(0.25),
    lc.Kurtosis(),
    lc.LinearTrend(),
    lc.MagnitudePercentageRatio(quantile_numerator=0.4, quantile_denominator=0.05),
    lc.MaximumSlope(),
    lc.Mean(),
    lc.Median(),
    lc.MedianAbsoluteDeviation(),
    lc.PercentAmplitude(),
    lc.Skew(),
    lc.StandardDeviation(),
    lc.StetsonK(),
)

# Features to extract using the cesium-package
cesium_features = [
    "freq1_freq",
    "freq2_freq",
    "stetson_j",
    "flux_percentile_ratio_mid20",
    "flux_percentile_ratio_mid50",
    "flux_percentile_ratio_mid80",
]


# Get the features for a light curve.
def get_LC_features(targets, times):
    """
    Get the light-curve features for a single light curve.

    """
    if len(targets) < 2 or np.all(targets == targets[0]):
        # If the light curve is too short just get the mean and median
        LC_features = np.zeros(21)
        LC_features[14] = np.mean(targets)
        LC_features[15] = np.median(targets)
    else:
        LC_features = LC_extractor(times, targets, sorted=True, check=False)
    return LC_features


def get_cesium_features(times, targets, features_to_use):
    """
    Get Cesium features for a single light curve.
    This function is designed to be called in parallel.
    """
    # Suppress warnings in each parallel worker process
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            fset = featurize.featurize_time_series(
                times=[times],
                values=[targets],
                errors=None,
                features_to_use=features_to_use,
            )
            return fset.values[0]
        except Exception:
            return np.zeros(len(features_to_use))


def downsample_batch(targets_list, times_list, max_points=500):
    """
    Downsamples arrays in the list that exceed max_points.
    Returns new lists with downsampled data.
    """
    ds_targets = []
    ds_times = []

    for t, m in zip(targets_list, times_list):
        n_obs = len(t)
        if n_obs > max_points:
            # 1. Pick random indices
            idx = np.random.choice(n_obs, max_points, replace=False)
            # 2. Sort indices to ensure time remains chronological!
            idx.sort()
            ds_targets.append(t[idx])
            ds_times.append(m[idx])
        else:
            ds_targets.append(t)
            ds_times.append(m)

    return ds_targets, ds_times


if __name__ == "__main__":
    for split in ["test", "train"]:
        input_path = f"data/raw/star/star-X-{split}.parquet"
        output_path = f"data/embeddings/star/CUSTOM-star-{split}.parquet"

        if not os.path.exists(input_path):
            print(f"Skipping {input_path}")
            continue

        print(f"\nLoading {input_path}...")
        df = pd.read_parquet(input_path)
        n = df.shape[0]

        all_embeddings = []

        for c in ["r", "g"]:
            print(f"Processing {c} wave...")

            # Pre-calculate lists to avoid pandas overhead in loop
            targets = list(map(np.asarray, df[f"{c}_target"]))
            times = list(map(np.asarray, df[f"{c}_mjd"]))

            # Storage for this color channel
            channel_features = []

            # Process in BATCHES
            for i in tqdm(range(0, n, BATCH_SIZE)):
                batch_targets = targets[i : i + BATCH_SIZE]
                batch_times = times[i : i + BATCH_SIZE]
                current_batch_len = len(batch_targets)

                # Light curve feature extraction
                lc_feats = np.array(
                    [get_LC_features(t, m) for t, m in zip(batch_targets, batch_times)]
                )

                #
                cesium_feats = np.zeros((current_batch_len, len(cesium_features)))

                # Identify valid rows (length >= 3)
                valid_indices = [j for j, t in enumerate(batch_targets) if len(t) >= 3]

                if valid_indices:
                    # Extract valid data
                    valid_targets = [batch_targets[j] for j in valid_indices]
                    valid_times = [batch_times[j] for j in valid_indices]

                    # Apply Downsampling to accelerate cesium feature extraction
                    ds_targets, ds_times = downsample_batch(
                        valid_targets, valid_times, max_points=CESIUM_MAX_POINTS
                    )

                    # Compute cesium features in parallel to speed up calculations
                    results = Parallel(n_jobs=-1, backend="loky")(
                        delayed(get_cesium_features)(t, m, cesium_features)
                        for t, m in zip(ds_times, ds_targets)
                    )
                    cesium_feats[valid_indices] = np.array(results)

                # Stack and store
                batch_combined = np.hstack((lc_feats, cesium_feats))
                channel_features.append(batch_combined)
                channel_features.append(lc_feats)

            # Stack all batches for this color
            col_embeddings = np.vstack(channel_features)
            all_embeddings.append(col_embeddings)

        print("Saving embeddings...")
        final_df = np.hstack(all_embeddings)

        cols = [f"emb_{i}" for i in range(final_df.shape[1])]
        emb_df = pd.DataFrame(final_df, columns=cols)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        emb_df.to_parquet(output_path, compression="zstd")
        print(f"Saved to {output_path}")
