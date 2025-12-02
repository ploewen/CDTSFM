# Purpose:
# Makes FinCast embeddings for StarEmbed data.

# Pre-requisites:
# - Requires running process-star.py, fincast-setup.sh,

# Authors:
# - Code written by Philip Loewen

# Reference:
# Z. Zhu, H. Chen, Q. Qu, and V. Chung,
# “FINCAST: a foundation model for Financial Time-Series Forecasting,”
# arXiv (Cornell University), Aug. 2025, doi: 10.48550/arxiv.2508.19609.

import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
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
# CPU workers for data loading (set to 0 for debugging, 4+ for speed)
NUM_WORKERS = 4
# Rows to read from Parquet at once (RAM dependent)
CHUNK_SIZE = 5000
# Rows to feed to GPU at once (VRAM dependent)
BATCH_SIZE = 16
# Patch length for the Transformer
SEQUENCE_LENGTH = 32


class PatchedTimeSeriesEmbeddingGenerator(PatchedTimeSeriesDecoder_MOE):
    """
    A time-series embedding generator adapted from the PatchedTimeSeriesDecoder_MOE class.
    Modifies forward pass to return embeddings before the prediction head.
    """

    def __init__(self, config: FFMConfig):
        """
        Initializes the PatchedTimeSeriesEmbeddingGenerator.

        Args:
            config (FFMConfig): Configuration object for the FFM model.
        """
        super().__init__(config)

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate time-series embeddings.

        Args:
            input_ts (torch.Tensor): Input time series tensor.
            input_padding (torch.LongTensor): Padding tensor for the input.
            freq (torch.Tensor): Frequency tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the model output (embeddings).
        """
        model_input, patched_padding, _, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )

        f_emb = self.freq_emb(freq).unsqueeze(1)
        model_input += f_emb

        model_output, _ = self.stacked_transformer(model_input, patched_padding)

        return model_output


class RaggedDataset(Dataset):
    """
    Optimized Dataset that holds references to numpy arrays instead of
    converting to Torch tensors immediately.
    """

    def __init__(self, df):
        """
        Initializes the RaggedDataset.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
        """
        self.data = df.iloc[:, 0].values
        self.indices = df.index.tolist()

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the values and index for the item.
        """
        # Convert to tensor only when requested by DataLoader
        return {
            "vals": torch.tensor(self.data[idx], dtype=torch.float32),
            "index": self.indices[idx],
        }


def pad_sequence(batch):
    """
    Pads a BATCH of ragged sequences to the longest sequence in the batch,
    ensuring the length is a multiple of SEQUENCE_LENGTH (32).

    Args:
        batch (list): A list of dictionaries, where each dictionary contains
            "vals" (torch.Tensor) and "index" (int).

    Returns:
        dict: A dictionary containing the padded sequences ("vals") and their
            corresponding original indices ("index").
    """
    # Separate values and indices
    sequences = [item["vals"] for item in batch]
    indices = [item["index"] for item in batch]

    # Find max length in this batch
    max_len = max(s.size(0) for s in sequences)

    # Calculate target length (must be multiple of 32 for the patcher)
    remainder = max_len % SEQUENCE_LENGTH
    if remainder != 0:
        target_len = max_len + (SEQUENCE_LENGTH - remainder)
    else:
        target_len = max_len

    # Pad and Stack
    padded_batch = []
    for seq in sequences:
        pad_size = target_len - seq.size(0)
        if pad_size > 0:
            # Pad at the BEGINNING (left padding)
            pad = torch.zeros(pad_size, dtype=seq.dtype)
            padded_seq = torch.cat((pad, seq))
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)

    # Stack into (Batch_Size, Time_Steps)
    return {"vals": torch.stack(padded_batch), "index": indices}


def make_embeddings(data, batch_size, device, model):
    """
    Generates and saves FinCast embeddings for a given dataset.

    Args:
        data (str): The name of the dataset (e.g., "star").
        batch_size (int): The number of rows to feed to the GPU at once.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            run the model.
        model (torch.nn.Module): The embedding model that accepts time-series
            values, a mask, and frequencies as input.
    """
    for split in ["test", "train"]:
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
            writer = make_chunk_embeddings(
                batch_chunk,
                batch_size,
                chunk_idx,
                model,
                output_path,
                cols,
                writer,
                device,
            )

        if writer:
            writer.close()

        print(f"Successfully saved to {output_path}")
        gc.collect()


def get_data(data, split):
    """
    Constructs input and output paths and prepares the Parquet file for reading.

    Args:
        data (str): The name of the dataset (e.g., "star").
        split (str): The data split (e.g., "train", "test").

    Returns:
        tuple: A tuple containing:
            - output_path (str): The path to the output embeddings file.
            - parquet_file (pyarrow.parquet.ParquetFile): The Parquet file object for the input data.
    """
    print(f"\nProcessing {data}-{split}...")
    input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
    output_path = f"data/embeddings/{data}/FinCast-{data}-{split}.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    parquet_file = pq.ParquetFile(input_path)
    return output_path, parquet_file


def make_chunk_embeddings(
    batch_chunk, batch_size, chunk_idx, model, OUTPUT_PATH, cols, writer, device
):
    """
    Processes a chunk of data to generate and save embeddings.

    Args:
        batch_chunk (pyarrow.RecordBatch): A chunk of data from a Parquet file.
        batch_size (int): The number of rows to feed to the GPU at once.
        chunk_idx (int): The index of the current chunk.
        model (torch.nn.Module): The embedding model.
        OUTPUT_PATH (str): The path to the output Parquet file.
        cols (list[str] or None): Column names for the embeddings. None for the first batch.
        writer (pyarrow.parquet.ParquetWriter or None): The Parquet writer object. None for the first batch.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            run the model.

    Returns:
        pyarrow.parquet.ParquetWriter: The Parquet writer instance.
    """
    # Convert only this chunk to Pandas
    X_df_chunk = batch_chunk.to_pandas()

    dataset = RaggedDataset(X_df_chunk)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_sequence,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}", leave=False):
            writer = make_batch_embeddings(
                device, model, OUTPUT_PATH, writer, cols, batch
            )

    return writer


def make_batch_embeddings(device, model, OUTPUT_PATH, writer, cols, batch):
    """Generates and writes embeddings for a single batch to a Parquet file.

    Args:
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            run the model.
        model (torch.nn.Module): The embedding model that accepts time-series
            values, a mask, and frequencies as input.
        OUTPUT_PATH (str): The path to the output Parquet file.
        writer (pyarrow.parquet.ParquetWriter or None): The writer object for the
            Parquet file. Pass None for the first batch.
        cols (list[str] or None): A list of column names for the embeddings.
            Pass None for the first batch to auto-generate them.
        batch (dict): A dictionary containing the batch data with keys:
            - "vals" (torch.Tensor): A tensor of shape (Batch_Size, Time_Steps)
              containing the time-series values. Padded values must be 0.
            - "index": An array-like object containing the unique identifiers
              for each time series in the batch, used for the output index.

    Returns:
        pyarrow.parquet.ParquetWriter: The Parquet writer instance, which should be
            passed to the next call to continue writing to the same file.
    """
    # batch["vals"] is now (Batch_Size, Time_Steps)
    batch_vals = batch["vals"].to(device, dtype=torch.float32, non_blocking=True)

    # Mask: 1 for data, 0 for padding
    batch_mask = (batch_vals != 0).type_as(batch_vals)

    batch_indices = batch["index"]

    # Frequencies: (Batch_Size,)
    batch_freq = torch.zeros(batch_vals.shape[0], dtype=torch.long, device=device)

    # Inference
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            embeddings = model(batch_vals, batch_mask, batch_freq)
    else:
        embeddings = model(batch_vals, batch_mask, batch_freq)

    # Aggregate: Mean over time dimension -> (Batch_Size, Hidden_Dim)
    aggregated_embeddings = embeddings.mean(dim=1).cpu()
    final_array = aggregated_embeddings.numpy()

    # Create column names on the first pass
    if writer is None and cols is None:
        cols = [f"emb_{j}" for j in range(final_array.shape[1])]

    emb_df = pd.DataFrame(final_array, columns=cols, index=batch_indices)
    table = pa.Table.from_pandas(emb_df)

    # Initialize writer on the first pass
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="zstd")

    writer.write_table(table=table)
    return writer


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
    else:
        print("WARNING: CUDA not found.")
        device = torch.device("cpu")

    embedder = PatchedTimeSeriesEmbeddingGenerator(config)

    print("Loading weights...")
    state_dict = torch.load("weights/v1.pth", map_location="cpu", weights_only=True)
    embedder.load_state_dict(state_dict, strict=True)
    embedder.to(device)
    embedder.eval()

    # --- OPTIMIZATION: Compile Model ---
    try:
        print("Compiling model for faster inference...")
        embedder = torch.compile(embedder)
    except Exception as e:
        print(f"Compilation skipped (not supported/failed): {e}")

    # Run the main process
    make_embeddings("star", batch_size=BATCH_SIZE, device=device, model=embedder)
