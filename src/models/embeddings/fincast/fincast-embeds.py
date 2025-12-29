# Purpose:
# Makes FinCast embeddings for S&P 500, ESC 50, PTB XL Datasets.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py,
#   fincast-setup.sh

# Authors:
# - Code written by Philip Loewen

# Reference:
# Z. Zhu, H. Chen, Q. Qu, and V. Chung,
# “FINCAST: a foundation model for Financial Time-Series Forecasting,”
# arXiv (Cornell University), Aug. 2025, doi: 10.48550/arxiv.2508.19609.

import torch
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import gc

# Import your model components
from ffm.pytorch_patched_decoder_MOE import (
    PatchedTimeSeriesDecoder_MOE,
    FFMConfig,
    create_quantiles,
)

# --- CONFIGURATION ---
# CRITICAL FOR SPEED:
WINDOW_SIZE = 2048  # Slice long audio into 2048-step windows
STRIDE = 2048  # No overlap between windows
BATCH_SIZE = 64  # Batch size for windows
NUM_WORKERS = 0  # Keep at 0 to avoid process overhead
CHUNK_SIZE = 500  # Number of original files to load at once


class PatchedTimeSeriesEmbeddingGenerator(PatchedTimeSeriesDecoder_MOE):
    def __init__(self, config: FFMConfig):
        super().__init__(config)

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_input, patched_padding, _, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )

        f_emb = self.freq_emb(freq).unsqueeze(1)
        model_input += f_emb

        model_output, _ = self.stacked_transformer(model_input, patched_padding)

        return model_output


class WindowedTimeSeriesDataset(Dataset):
    """
    Slices long time-series rows into smaller sliding windows.
    """

    def __init__(self, df, window_size=2048, stride=2048):
        self.window_size = window_size
        self.stride = stride
        self.original_indices = df.index.tolist()
        self.np_data = df.values.astype(np.float32)

        self.samples = []
        n_rows, n_cols = self.np_data.shape

        for row_idx in range(n_rows):
            original_idx = self.original_indices[row_idx]

            # Slide window across the row
            for start in range(0, n_cols, stride):
                end = min(start + window_size, n_cols)
                length = end - start

                # Skip tiny hanging tails (< 25% of window)
                if length < (window_size // 4):
                    continue

                self.samples.append(
                    {
                        "row_idx": row_idx,
                        "original_index": original_idx,
                        "start": start,
                        "end": end,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        row_idx = sample_info["row_idx"]
        s, e = sample_info["start"], sample_info["end"]

        window = self.np_data[row_idx, s:e]

        if len(window) < self.window_size:
            pad_amt = self.window_size - len(window)
            window = np.concatenate([window, np.zeros(pad_amt, dtype=np.float32)])

        return {
            "vals": torch.tensor(window, dtype=torch.float32),
            "index": sample_info["original_index"],
        }


def make_embeddings(data, batch_size, device, model):
    for split in ["train", "test"]:
        try:
            output_path, parquet_file = get_data(data, split)
        except FileNotFoundError:
            print(f"File not found: {data}-{split}, skipping...")
            continue

        writer = None
        cols = None

        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows: {total_rows}. Processing in chunks of {CHUNK_SIZE}...")

        chunk_iterator = parquet_file.iter_batches(batch_size=CHUNK_SIZE)

        for chunk_idx, batch_chunk in enumerate(chunk_iterator):
            writer, cols = make_chunk_embeddings(
                batch_chunk, batch_size, chunk_idx, model, output_path, cols, writer
            )

        if writer:
            writer.close()

        print(f"Successfully saved to {output_path}")
        gc.collect()


def get_data(data, split):
    print(f"\nProcessing {data}-{split}...")
    input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
    output_path = f"data/embeddings/{data}/FinCast-{data}-{split}.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    parquet_file = pq.ParquetFile(input_path)
    return output_path, parquet_file


def make_chunk_embeddings(
    batch_chunk, batch_size, chunk_idx, model, OUTPUT_PATH, cols, writer
):
    # 1. Convert to Pandas
    X_df_chunk = batch_chunk.to_pandas()

    # 2. Setup Windowed Dataset
    dataset = WindowedTimeSeriesDataset(
        X_df_chunk, window_size=WINDOW_SIZE, stride=STRIDE
    )

    # 3. Setup Loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    chunk_embeddings = []
    chunk_indices = []

    # 4. Inference Loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}", leave=False):
            batch_vals = batch["vals"].to(device, non_blocking=True)
            batch_mask = (batch_vals != 0).type_as(batch_vals)

            batch_freq = torch.zeros(
                batch_vals.shape[0], dtype=torch.long, device=device
            )

            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = model(batch_vals, batch_mask, batch_freq)
            else:
                embeddings = model(batch_vals, batch_mask, batch_freq)

            # Mean pooling over TIME dimension (for each window)
            aggregated = embeddings.mean(dim=1).cpu().numpy()
            chunk_embeddings.append(aggregated)

            # Collect indices (repeated for multiple windows of same file)
            b_idx = batch["index"]
            if isinstance(b_idx, torch.Tensor):
                b_idx = b_idx.tolist()
            chunk_indices.extend(b_idx)

    # 5. Aggregation and Writing
    if len(chunk_embeddings) > 0:
        full_chunk_array = np.vstack(chunk_embeddings)

        if cols is None:
            cols = [f"emb_{j}" for j in range(full_chunk_array.shape[1])]

        # Create DataFrame with potentially duplicate indices
        emb_df = pd.DataFrame(full_chunk_array, columns=cols, index=chunk_indices)

        # Ensure index is string
        emb_df.index = emb_df.index.astype(str)

        # --- THE FIX: AGGREGATE BY INDEX ---
        # Group by the index (filename) and take the mean of all windows
        averaged_df = emb_df.groupby(emb_df.index).mean()

        # Convert to Table
        table = pa.Table.from_pandas(averaged_df)

        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="zstd")

        writer.write_table(table=table)

    # Cleanup
    del X_df_chunk, dataset, dataloader, chunk_embeddings
    if "full_chunk_array" in locals():
        del full_chunk_array
    gc.collect()

    return writer, cols


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    config = FFMConfig(
        num_layers=50,
        num_heads=16,
        num_kv_heads=16,
        hidden_size=1280,
        intermediate_size=1280,
        head_dim=80,
        rms_norm_eps=1e-6,
        patch_len=32,
        horizon_len=128,
        quantiles=create_quantiles(),
        pad_val=1123581321.0,
        tolerance=1e-6,
        dtype="bfloat32",
        use_positional_embedding=False,
        num_experts=4,
        gating_top_n=2,
        threshold_train=0.2,
        threshold_eval=0.2,
    )

    print("Initializing model...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS.")
    else:
        print("Using CPU.")
        device = torch.device("cpu")

    embedder = PatchedTimeSeriesEmbeddingGenerator(config)

    print("Loading weights...")
    state_dict = torch.load("weights/v1.pth", map_location="cpu", weights_only=True)
    embedder.load_state_dict(state_dict, strict=True)
    embedder.to(device)
    embedder.eval()

    make_embeddings("ptb-xl", batch_size=BATCH_SIZE, device=device, model=embedder)
    make_embeddings("esc-50", batch_size=BATCH_SIZE, device=device, model=embedder)
    make_embeddings("SP-500", batch_size=256, device=device, model=embedder)
