# Purpose:
# Makes creates classifications for all datasets.
#


# Pre-requisites:
# - Requires running all embeddings scripts

# Authors:
# - Code written by Philip Loewen


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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_pred):
    """
    Helper to calculate Acc, Prec, Rec, F1 (weighted average).
    Returns a dictionary.
    """
    # Move tensors to cpu numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"Acc": acc, "Prec": prec, "Rec": rec, "F1": f1}


def evaluate(X_train, y_train, X_test, y_test, n_components=32):
    """
    Evaluates LogReg, kNN, and Centroid on PCA-reduced features.
    Returns a flat dictionary of metrics.
    """
    # Define and Fit Common Preprocessor
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

    clf_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf_lr.fit(X_train_pca, y_train)
    y_pred_lr = clf_lr.predict(X_test_pca)

    metrics_lr = compute_metrics(y_test, y_pred_lr)
    for k, v in metrics_lr.items():
        results[f"LogReg_{k}"] = v

    normalizer = Normalizer(norm="l2")
    X_train_norm = normalizer.fit_transform(X_train_pca)
    X_test_norm = normalizer.transform(X_test_pca)

    clf_knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf_knn.fit(X_train_norm, y_train)
    y_pred_knn = clf_knn.predict(X_test_norm)

    metrics_knn = compute_metrics(y_test, y_pred_knn)
    for k, v in metrics_knn.items():
        results[f"kNN_{k}"] = v

    clf_nc = NearestCentroid()
    clf_nc.fit(X_train_norm, y_train)
    y_pred_nc = clf_nc.predict(X_test_norm)

    metrics_nc = compute_metrics(y_test, y_pred_nc)
    for k, v in metrics_nc.items():
        results[f"Centroid_{k}"] = v

    print(
        f"    [Lightweight] LogReg Acc: {metrics_lr['Acc']:.3f} | kNN Acc: {metrics_knn['Acc']:.3f}"
    )
    return results


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


def train_mlp_gpu_multiseed(
    X_train_raw, y_train, X_test_raw, y_test, device, epochs=50, batch_size=256
):
    """
    Trains an MLP 3 times with different seeds.
    Returns Mean and Std for Acc, Prec, Rec, F1.
    """
    # Seeds for reproducibility
    seeds = [0, 1, 2]

    # Preprocessing (Done once)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    # Prepare storage for metrics across seeds
    run_metrics = {"Acc": [], "Prec": [], "Rec": [], "F1": []}

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    for seed in seeds:
        # Set seed for this run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create DataLoader (Re-instantiate to respect seed if shuffling needed, though dataset is fixed here)
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = MLPProbe(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)

            # Compute metrics for this seed
            m = compute_metrics(y_test_t, predicted)

            for k in run_metrics:
                run_metrics[k].append(m[k])

    # Compute Mean and Std
    final_results = {}
    for k in run_metrics:
        final_results[f"MLP_{k}_Mean"] = np.mean(run_metrics[k])
        final_results[f"MLP_{k}_Std"] = np.std(run_metrics[k])

    print(
        f"    [MLP] Avg Accuracy: {final_results['MLP_Acc_Mean']:.4f} (+/- {final_results['MLP_Acc_Std']:.4f})"
    )
    return final_results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    datasets = ["SP-500", "star"]
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

        # Load Labels
        try:
            y_df_train = pd.read_parquet(
                f"/kaggle/input/project/{data}-y-train.parquet"
            )
            y_df_test = pd.read_parquet(f"/kaggle/input/project/{data}-y-test.parquet")

            if data == "ptb-xl":

                def get_label(x):
                    if len(x) > 0:
                        return x[0]
                    else:
                        return "OTHER"

                y_df_train = [get_label(x) for x in y_df_train["diagnostic_superclass"]]
                y_df_test = [get_label(x) for x in y_df_test["diagnostic_superclass"]]

            y_train_raw = np.array(y_df_train).reshape(-1)
            y_test_raw = np.array(y_df_test).reshape(-1)

            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test = le.transform(y_test_raw)
        except Exception as e:
            print(f"Error loading labels for {data}: {e}")
            continue

        if y_train.dtype == "object" or y_test.dtype == "object":
            print("    ! Object type detected. Coercing to int...")
            y_test = list(map(int, y_test))
            y_train = list(map(int, y_train))

        # Encoding for ptb-xl if needed (omitted here for brevity, keep your original logic)

        for model in models:
            print(f"  > Model: {model}")

            try:
                # Load Features
                X_df_train = pd.read_parquet(
                    f"/kaggle/input/embeddings/embeddings/{data}/{model}-{data}-train.parquet"
                )
                X_train = np.array(X_df_train)

                X_df_test = pd.read_parquet(
                    f"/kaggle/input/embeddings/embeddings/{data}/{model}-{data}-test.parquet"
                )
                X_test = np.array(X_df_test)

                # Fix Shape Mismatch (RANDOM)
                if len(X_train) != len(y_train):
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

                # Fix NaNs
                if np.isnan(X_train).any() or np.isnan(X_test).any():
                    print("    ! NaNs detected after coercion. Imputing with 0...")
                    imputer = SimpleImputer(strategy="constant", fill_value=0)
                    X_train = imputer.fit_transform(X_train)
                    X_test = imputer.transform(X_test)

                # Run Lightweight Evaluation (Returns dictionary of metrics)
                n_comp = min(50, X_train.shape[1])
                results_light = evaluate(
                    X_train, y_train, X_test, y_test, n_components=n_comp
                )

                # Run MLP Evaluation (Returns dictionary of Mean/Std metrics)
                results_mlp = train_mlp_gpu_multiseed(
                    X_train, y_train, X_test, y_test, device, epochs=15
                )

                # Merge all results into one record
                record = {"Embedding": model}
                record.update(results_light)
                record.update(results_mlp)

                dataset_results.append(record)

            except Exception as e:
                print(f"    ! CRITICAL FAILURE on {data}-{model}: {e}")
                # Append empty record with correct keys set to 0 to preserve table structure?
                # Usually better to just skip or log error. For now, logging error.

        # Generate table
        results_df = pd.DataFrame(dataset_results)

        if not results_df.empty:
            cols = list(results_df.columns)
            if "Embedding" in cols:
                cols.remove("Embedding")
                cols = ["Embedding"] + sorted(cols)  # Alphabetical sort of metrics

            results_df = results_df[cols]
            results_df = results_df.round(4)

            print(results_df.to_string(index=False))

            # Save individual CSV for this dataset
            filename = f"data/results/results_{data}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nSaved to {filename}")
        else:
            print("No results generated for this dataset.")

        print("\n\n")
