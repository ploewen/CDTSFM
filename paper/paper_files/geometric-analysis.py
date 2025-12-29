import os
import glob
import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_ROOT = "data/embeddings"
TARGET_SPLIT = "test"  # Audit the test set to see the zero-shot manifold
# Columns to explicitly ignore if found (add your specific label/metadata columns here)
IGNORE_COLS = {"label", "target", "class", "y", "id", "date", "time", "index", "split"}


class GeometricAudit:
    def __init__(self):
        self.results = []

    def _estimate_intrinsic_dimension(self, X):
        """
        Estimates Intrinsic Dimension using the Two-NN algorithm (Facco et al., 2017).
        Robust to high-dimensions.
        """
        N = X.shape[0]
        if N < 50:
            return 0  # Too small for stable estimate

        # Compute distances to 2 nearest neighbors
        # n_neighbors=3 because index 0 is the point itself
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Ratio of 2nd NN dist to 1st NN dist
        r1 = distances[:, 1]
        r2 = distances[:, 2]

        # Filter duplicates/zeros to avoid division errors
        mask = r1 > 0
        r1, r2 = r1[mask], r2[mask]

        if len(r1) == 0:
            return 0

        mu = r2 / r1

        # ID Estimate formula: ID = N / sum(log(mu))
        id_est = len(mu) / np.sum(np.log(mu))
        return id_est

    def _calculate_anisotropy(self, X, n_pairs=2000):
        """
        Calculates the 'Cone Effect' (Anisotropy).
        Avg Cosine Similarity of random pairs.
        > 0.9 = High Anisotropy (Collapsed Cone).
        ~ 0.0 = Isotropic (Spherical/Noise).
        """
        N = X.shape[0]
        if N < 2:
            return 0

        # Cap n_pairs to actual data size
        n_pairs = min(n_pairs, N)

        # Sample random indices
        idx1 = np.random.choice(N, n_pairs)
        idx2 = np.random.choice(N, n_pairs)

        # Compute Cosine Sim (sklearn handles normalization)
        sims = cosine_similarity(X[idx1], X[idx2]).diagonal()
        return np.mean(sims)

    def _calculate_linearity(self, X, threshold=0.95):
        """
        PCA Audit. Returns # of components needed to explain 95% variance.
        """
        # Limit components to min(N, D) or 512
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # Find index where variance > threshold
        matches = np.where(cumsum >= threshold)[0]
        if len(matches) > 0:
            d_95 = matches[0] + 1
        else:
            d_95 = n_components

        return d_95

    def audit_file(self, dataset_name, model_name, X):
        print(f"Auditing {dataset_name} - {model_name} (Shape: {X.shape})...")

        # Intrinsic Dimension
        id_val = self._estimate_intrinsic_dimension(X)

        # Anisotropy
        anisotropy = self._calculate_anisotropy(X)

        # Linearity (PCA dim)
        d95 = self._calculate_linearity(X)

        self.results.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Intrinsic_Dim": round(id_val, 2),
                "Anisotropy": round(anisotropy, 4),
                "PCA_Dim_95": int(d95),
                "Embedding_Size": X.shape[1],
                "Sample_Size": X.shape[0],
            }
        )


def load_and_clean_parquet(filepath):
    """
    Loads parquet and tries to isolate the embedding features.
    Assumes numeric columns that are not in IGNORE_COLS are features.
    """
    try:
        # Using Polars for speed, converting to pandas/numpy for compatibility
        df = pl.read_parquet(filepath)

        # Identify non-feature columns (strings, dates, or ignored names)
        feature_cols = []
        for col in df.columns:
            # Check if column name suggests it's metadata
            if col.lower() in IGNORE_COLS:
                continue
            # Check dtype (we only want numeric)
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)

        # Selection
        data_df = df.select(feature_cols)

        # Convert to numpy array
        X = data_df.to_numpy()

        # Safety check: Infinite or NaN values
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"Warning: NaNs/Infs found in {filepath}. Replacing with 0.")
            X = np.nan_to_num(X)

        return X

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def main():
    auditor = GeometricAudit()

    # Path pattern: data/embeddings/{data}/{model}-{data}-{split}.parquet
    # We look for folders inside data/embeddings
    dataset_folders = glob.glob(os.path.join(DATA_ROOT, "*"))

    for folder in dataset_folders:
        if not os.path.isdir(folder):
            continue

        dataset_name = os.path.basename(folder)

        # Find all parquet files in this folder matching the split
        # Pattern: *{split}.parquet
        search_pattern = os.path.join(folder, f"*-{TARGET_SPLIT}.parquet")
        files = glob.glob(search_pattern)

        if not files:
            print(f"No '{TARGET_SPLIT}' files found for dataset: {dataset_name}")
            continue

        for filepath in files:
            filename = os.path.basename(filepath)

            # Parse Model Name
            # Expecting filename: "{model}-{data}-{split}.parquet"
            # We can split by '-'
            parts = filename.replace(".parquet", "").split("-")

            # Heuristic to find model name (everything before the dataset name)
            # This handles cases where model name has hyphens like 'wav2vec-2.0'
            try:
                # Find index of dataset_name in the parts
                # Note: dataset_name in folder path might differ slightly from filename data part
                # So we assume the LAST part is split, the SECOND TO LAST is data, rest is model
                if len(parts) >= 3:
                    model_name = "-".join(parts[:-2])
                else:
                    model_name = "unknown"
            except Exception:
                model_name = filename

            # Load Data
            X = load_and_clean_parquet(filepath)

            if X is not None and X.shape[1] > 1:
                auditor.audit_file(dataset_name, model_name, X)

    # Compile Results
    if auditor.results:
        results_df = pd.DataFrame(auditor.results)
        results_df = results_df.sort_values(by=["Dataset", "Intrinsic_Dim"])

        print("\n=== GEOMETRIC AUDIT RESULTS ===")
        print(results_df)

        # Save to CSV for your paper
        results_df.to_csv("geometric_audit_results.csv", index=False)
        print("\nSaved results to 'geometric_audit_results.csv'")
    else:
        print("No results generated. Check your paths and file names.")


if __name__ == "__main__":
    main()
