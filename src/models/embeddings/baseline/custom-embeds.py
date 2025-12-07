# Purpose:
# Makes custom embeddings for S&P 500, ESC 50, PTB XL Datasets.
#
# Extract features using the ta package for S&P 500 data.
# Extract features using the librosa package for ESC 50 data.
# Extract features using the neurokit2 package for PTB XL data.


# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py

# Authors:
# - Code written by Philip Loewen

# Reference:
# B. McFee et al., “librosa/librosa: 0.11.0,” Zenodo (CERN European Organization for
# Nuclear Research), Mar. 2025, doi: 10.5281/zenodo.15006942.
#
# D. Makowski et al, (2021). "NeuroKit2: A Python toolbox for neurophysiological signal processing."
# Behavior Research Methods, 53(4), 1689-1696. https://doi.org/10.3758/s13428-020-01516-y

import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm
import ta.momentum
import ta.trend
import ta.volatility
import librosa
import neurokit2 as nk
from scipy.stats import skew, kurtosis

# Suppress warnings from libraries dealing with short/noisy signals
warnings.filterwarnings("ignore")

# Congig
SR_AUDIO = 16000  # Hz for Audio
SR_ECG = 100  # Hz for PTB-XL


def get_general_statistics(row_data):
    """
    Calculates 13 fundamental invariants:
    - Distribution: Mean, Std, Min, Max, Median, PTP, Skew, Kurtosis
    - Energy: RMS, Area Under Curve (AUC)
    - Dynamics: ZCR, MCR, Autocorrelation, Dominant Frequency
    """
    x = np.nan_to_num(np.array(row_data, dtype=float))

    # Basic statistics
    mean = np.mean(x)
    std = np.std(x)
    minimum = np.min(x)
    maximum = np.max(x)
    median = np.median(x)
    ptp = maximum - minimum

    # Shape
    skew_val = skew(x)
    kurt_val = kurtosis(x)

    # Energy
    # Root Mean Square
    rms = np.sqrt(np.mean(x**2))
    # AUC: Total absolute area
    auc = np.sum(np.abs(x))

    # Dynamics & Frequency
    # Zero Crossing Rate
    zcr = ((x[:-1] * x[1:]) < 0).sum() / (len(x) + 1e-6)

    # Mean Crossing Rate
    centered = x - mean
    mcr = ((centered[:-1] * centered[1:]) < 0).sum() / (len(x) + 1e-6)

    # Autocorrelation (Lag 1)
    if std > 1e-9:
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    else:
        ac1 = 0.0  # Constant signal

    # Dominant Frequency
    try:
        fft = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x))
        dom_freq = freqs[np.argmax(np.abs(fft))]
    except Exception:
        dom_freq = 0.0

    return [
        mean,
        std,
        minimum,
        maximum,
        median,
        ptp,
        skew_val,
        kurt_val,
        rms,
        auc,
        zcr,
        mcr,
        ac1,
        dom_freq,
    ]


def get_finance_features(row_data):
    """Technical Analysis (TA) for Financial Data."""
    s = pd.Series(row_data)
    features = []
    try:
        # RSI (Momentum)
        features.append(ta.momentum.rsi(s, window=14).iloc[-1])
        # MACD (Trend)
        features.append(ta.trend.macd_diff(s).iloc[-1])
        # Bollinger Bands (Volatility)
        bb_h = ta.volatility.bollinger_hband(s, window=20).iloc[-1]
        bb_l = ta.volatility.bollinger_lband(s, window=20).iloc[-1]
        features.append(bb_h - bb_l)
        # Simple Return (Growth)
        features.append((s.iloc[-1] - s.iloc[0]) / (s.iloc[0] + 1e-6))
    except Exception:
        features = [0.0] * 4
    return features


def get_physio_features(row_data):
    """HRV metrics for ECG Data using NeuroKit2."""
    try:
        # Clean and find R-peaks
        clean_sig = nk.ecg_clean(row_data, sampling_rate=SR_ECG)
        peaks, _ = nk.ecg_peaks(clean_sig, sampling_rate=SR_ECG)

        # Calculate HRV (Time Domain only for speed/robustness on short segments)
        hrv = nk.hrv_time(peaks, sampling_rate=SR_ECG)

        # Extract key metrics
        return [
            hrv["HRV_RMSSD"].values[0],
            hrv["HRV_MeanNN"].values[0],
            hrv["HRV_SDNN"].values[0],
            hrv["HRV_pNN50"].values[0],
        ]
    except Exception:
        # Fallback if signal is too noisy/short for R-peak detection
        return [0.0] * 4


def get_audio_features(row_data):
    """Spectral Texture for Audio Data using Librosa."""
    x = np.array(row_data, dtype=float)
    # Pad short audio
    if len(x) < 2048:
        x = np.pad(x, (0, 2048 - len(x)))

    try:
        # MFCCs (Timbre)
        mfcc = np.mean(librosa.feature.mfcc(y=x, sr=SR_AUDIO, n_mfcc=13), axis=1)
        # Spectral Centroid (Brightness)
        cent = np.mean(librosa.feature.spectral_centroid(y=x, sr=SR_AUDIO))
        # Spectral Contrast (Peaks/Valleys)
        cont = np.mean(librosa.feature.spectral_contrast(y=x, sr=SR_AUDIO), axis=1)

        return np.hstack([mfcc, cent, cont])
    except Exception:
        return np.zeros(21)  # 13 MFCC + 1 Cent + 7 Contrast


def process_dataset(data):
    for split in ["train", "test"]:
        input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
        output_path = f"data/embeddings/{data}/CUSTOM-{data}-{split}.parquet"

        if not os.path.exists(input_path):
            print(f"Skipping {input_path} (Not found)")
            continue

        print(f"\nProcessing {data}-{split}...")
        df = pd.read_parquet(input_path)
        X_iterable = df.values

        # Determine Domain Function
        if "SP-500" in data:
            domain_func = get_finance_features
            print(" -> Strategy: Finance (TA)")
        elif "ptb-xl" in data:
            domain_func = get_physio_features
            print(" -> Strategy: Physiology (NeuroKit2)")
        elif "esc-50" in data.lower():
            domain_func = get_audio_features
            print(" -> Strategy: Audio (Librosa)")
        else:
            print(" -> Strategy: Default (General Stats Only)")

            def domain_func(x):
                return []

        all_features = []

        # Loop with progress bar
        for row in tqdm(X_iterable, desc=f"Extracting {data}"):
            # Clean Data
            # Ensure it is a flat array of floats
            try:
                row_clean = np.array(row, dtype=float)
                row_clean = np.nan_to_num(row_clean)
            except Exception:
                # Handle edge case where row might be a list wrapped in an array
                row_clean = np.nan_to_num(np.array(list(row), dtype=float))

            # Get Domain Features
            d_feats = domain_func(row_clean)

            # Get General Statistics
            g_feats = get_general_statistics(row_clean)

            # Stack features
            combined = np.hstack([d_feats, g_feats])
            all_features.append(combined)

        # Finalize
        final_array = np.nan_to_num(np.array(all_features))

        print(f"Final Feature Vector Shape: {final_array.shape}")

        # Create DataFrame
        cols = [f"emb_{i}" for i in range(final_array.shape[1])]
        emb_df = pd.DataFrame(final_array, columns=cols)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        emb_df.to_parquet(output_path, compression="zstd")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    datasets = ["SP-500", "ptb-xl", "esc-50"]

    for ds in datasets:
        process_dataset(ds)
