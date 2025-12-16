import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def evaluate(X_train, y_train, X_test, y_test, n_components=32):
    """
    Evaluates LogReg, kNN, and Centroid on PCA-reduced features.
    """
    # 2. Define and Fit Common Preprocessor
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform PCA
    if X_train.shape[1] > n_components:
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
    else:
        X_train_pca = X_train_scaled
        X_test_pca = X_test_scaled

    results = {}

    # --- Head A: Logistic Regression ---
    clf_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf_lr.fit(X_train_pca, y_train)
    results["LogReg"] = clf_lr.score(X_test_pca, y_test)

    # --- Special Step for Distance Models ---
    normalizer = Normalizer(norm="l2")
    X_train_norm = normalizer.fit_transform(X_train_pca)
    X_test_norm = normalizer.transform(X_test_pca)

    # --- Head B: k-NN ---
    clf_knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf_knn.fit(X_train_norm, y_train)
    results["kNN"] = clf_knn.score(X_test_norm, y_test)

    # --- Head C: Nearest Centroid ---
    clf_nc = NearestCentroid()
    clf_nc.fit(X_train_norm, y_train)
    results["Centroid"] = clf_nc.score(X_test_norm, y_test)

    print(f"    [Lightweight] {results}")
    return results


# 1. Define the MLP Probe Module
class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPProbe, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_gpu(X_train, y_train, X_test, y_test, device, epochs=50, batch_size=256):
    """
    Trains an MLP on the raw embeddings using GPU.
    """
    # Scale data (Critical for convergence)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch Tensors and move to Device
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader for batching
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    model = MLPProbe(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)

    print(f"    [MLP] Accuracy: {accuracy:.4f}")
    return accuracy


# --- usage example ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    datasets = ["esc-50"]  # Focused on the problematic dataset
    models = [
        "W2V",
        "ASTRO",
        "FinCast",
        "ECGFounder",
        "TIMESFM",
        "MOIRAI",
        "CUSTOM",
        "RANDOM",
    ]

    for data in datasets:
        dataset_results = []
        print(f"\nProcessing Dataset: {data}")
        print("-" * 30)

        # 1. Load Labels
        y_df_train = pd.read_parquet(f"data/processed/{data}/{data}-y-train.parquet")
        y_train_raw = np.array(y_df_train).reshape(-1)

        y_df_test = pd.read_parquet(f"data/processed/{data}/{data}-y-test.parquet")
        y_test_raw = np.array(y_df_test).reshape(-1)

        # --- FIX: Ensure labels are 0-indexed integers ---
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)

        if y_train.dtype == "object" or y_test.dtype == "object":
            print("    ! Object type detected. Coercing to int...")
            y_test = torch.tensor(list(map(int, y_test)))
            y_train = torch.tensor(list(map(int, y_train)))

        # Encoding for ptb-xl if needed (omitted here for brevity, keep your original logic)

        for model in models:
            print(f"  > Model: {model}")

            try:
                # 2. Load Features
                X_df_train = pd.read_parquet(
                    f"data/embeddings/{data}/{model}-{data}-train.parquet"
                )
                X_train = np.array(X_df_train)

                X_df_test = pd.read_parquet(
                    f"data/embeddings/{data}/{model}-{data}-test.parquet"
                )
                X_test = np.array(X_df_test)

                # --- 3. CRITICAL DATA CLEANING BLOCK ---

                # B. Fix Shape Mismatch (RANDOM)
                if len(X_train) != len(y_train):
                    print(
                        f"    ! Shape mismatch: X={X_train.shape}, y={y_train.shape}. Trimming X..."
                    )
                    X_train = X_train[: len(y_train)]
                    X_test = X_test[: len(y_test)]

                if X_train.dtype == "object" or X_test.dtype == "object":
                    print("    ! Object type detected. Coercing to float32...")
                    X_train = (
                        pd.DataFrame(X_train)
                        .apply(pd.to_numeric, errors="coerce")
                        .to_numpy(dtype=np.float32)
                    )
                    X_test = (
                        pd.DataFrame(X_test)
                        .apply(pd.to_numeric, errors="coerce")
                        .to_numpy(dtype=np.float32)
                    )

                # D. Fix NaNs (Now that everything is float, we can safely impute)
                if np.isnan(X_train).any() or np.isnan(X_test).any():
                    print("    ! NaNs detected after coercion. Imputing with 0...")
                    imputer = SimpleImputer(strategy="constant", fill_value=0)
                    X_train = imputer.fit_transform(X_train)
                    X_test = imputer.transform(X_test)

                # E. Final Safety Cast
                X_train = torch.from_numpy(X_train)
                X_test = torch.from_numpy(X_test)

                # --- END CLEANING BLOCK ---

                # Run Lightweight Evaluation
                # Ensure n_components isn't larger than feature count
                n_comp = min(50, X_train.shape[1])
                results_light = evaluate(
                    X_train, y_train, X_test, y_test, n_components=n_comp
                )

                # Run MLP Evaluation
                mlp_acc = train_mlp_gpu(
                    X_train, y_train, X_test, y_test, device, epochs=50
                )

                record = {
                    "Embedding": model,
                    "LogReg": results_light["LogReg"],
                    "kNN": results_light["kNN"],
                    "Centroid": results_light["Centroid"],
                    "MLP": mlp_acc,
                }
                dataset_results.append(record)

            except Exception as e:
                print(f"    ! CRITICAL FAILURE on {data}-{model}: {e}")
                dataset_results.append(
                    {
                        "Embedding": model,
                        "LogReg": 0.0,
                        "kNN": 0.0,
                        "Centroid": 0.0,
                        "MLP": 0.0,
                    }
                )

        # --- Generate Table for THIS Dataset ---
        print("\n" + "=" * 60)
        print(f"RESULTS TABLE: {data.upper()}")
        print("=" * 60)

        results_df = pd.DataFrame(dataset_results)

        if not results_df.empty:
            # Round for display
            results_df = results_df.round(4)
            print(results_df.to_string(index=False))

            # Save individual CSV for this dataset
            filename = f"results_{data}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nSaved to {filename}")
        else:
            print("No results generated for this dataset.")

        print("\n\n")
