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

# CONFIG
NUM_WORKERS = 2
CHUNK_SIZE = 2500
SEQUENCE_LENGTH = 32
# We load 2500 rows at a time to save on memory


# This is the model we are going to use. It is adapted from the PatchedTimeSeriesDecoder_MOE
# class. The forward method has been modified to forego the postprocessing of embeddings
# which generated predictions.
class PatchedTimeSeriesEmbeddingGenerator(PatchedTimeSeriesDecoder_MOE):
    """
    A time-series embedding generator adapted from the PatchedTimeSeriesDecoder_MOE class.

    This class modifies the forward pass to return the model's internal embeddings
    before the final prediction head, making it suitable for feature extraction.
    """

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


# Dataset for dataloader
class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for handling time-series data, including padding.

    This dataset prepares time-series data for input into a model by handling
    padding to a specified `sequence_len` and converting data to `torch.float32`.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data.
        sequence_len (int, optional): The desired sequence length for padding.
                                     Defaults to 32.
    """

    def __init__(self, df, sequence_len=32):
        self.indices = df.index.tolist()
        self.vals = df.values.astype(np.float32)
        self.n_samples, self.m_features = self.vals.shape
        self.sequence_len = sequence_len
        _, r = divmod(self.m_features, sequence_len)
        self.pad_width = 0 if r == 0 else (sequence_len - r)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        row = self.vals[idx]
        observed = np.ones_like(row)

        if self.pad_width > 0:
            pad = np.zeros(self.pad_width, dtype=np.float32)
            padded_row = np.concatenate([pad, row])
            padded_mask = np.concatenate([pad, observed])
        else:
            padded_row = row
            padded_mask = observed

        return {
            "vals": torch.tensor(padded_row, dtype=torch.float32),
            "mask": torch.tensor(padded_mask, dtype=torch.float32),
            "index": self.indices[idx],
        }


def make_embeddings(data, batch_size, device, model):
    """
    Generates and saves FinCast embeddings for a given dataset.

    This function iterates through the 'test' and 'train' splits of the specified
    dataset, processes data in chunks, and generates embeddings using the provided
    model.

    Args:
        data (str): The name of the dataset (e.g., "star").
        batch_size (int): The batch size to use for the DataLoader.
        device (torch.device): The device (CPU or CUDA) to run the model on.
        model (PatchedTimeSeriesEmbeddingGenerator): The FinCast embedding model.
    """
    for split in ["test", "train"]:
        try:
            output_path, parquet_file = get_data(data, split)
        except FileNotFoundError:
            print(f"File not found: {data}-{split}, skipping...")
            continue

        writer = None
        cols = None

        # Iterate over the file in chunks to keep RAM usage low
        # iter_batches yields PyArrow RecordBatches
        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows: {total_rows}. Processing in chunks of {CHUNK_SIZE}...")

        chunk_iterator = parquet_file.iter_batches(batch_size=CHUNK_SIZE)

        for chunk_idx, batch_chunk in enumerate(chunk_iterator):
            make_chunk_embeddings(
                batch_chunk, batch_size, chunk_idx, model, output_path, cols, writer
            )

        if writer:
            writer.close()

        print(f"Successfully saved to {output_path}")
        gc.collect()


def get_data(data, split):
    """
    Constructs input and output file paths and opens a Parquet file for reading.

    Args:
        data (str): The name of the dataset.
        split (str): The data split (e.g., "test" or "train").

    Returns:
        Tuple[str, pq.ParquetFile]: A tuple containing the output file path
        and an opened ParquetFile object.

    Raises:
        FileNotFoundError: If the input Parquet file does not exist.
    """
    print(f"\nProcessing {data}-{split}...")
    input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
    output_path = f"data/embeddings/{data}/FinCast-{data}-{split}.parquet"

    # Ensure output directory exists and remove file if it already exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    # Open file handle without reading data
    parquet_file = pq.ParquetFile(input_path)
    return output_path, parquet_file


def make_chunk_embeddings(
    batch_chunk, batch_size, chunk_idx, model, OUTPUT_PATH, cols, writer
):
    """
    Processes a chunk of data to generate and save FinCast embeddings.

    Args:
        batch_chunk (pa.RecordBatch): A chunk of data as a PyArrow RecordBatch.
        batch_size (int): The batch size for the DataLoader.
        chunk_idx (int): The index of the current chunk.
        model (PatchedTimeSeriesEmbeddingGenerator): The FinCast embedding model.
        OUTPUT_PATH (str): The path to the output Parquet file.
        cols (list): A list of column names for the embeddings.
        writer (pq.ParquetWriter): The Parquet writer object.
    """
    # Convert only this chunk to Pandas
    X_df_chunk = batch_chunk.to_pandas()

    # Create Dataset/Loader for JUST this chunk
    dataset = TimeSeriesDataset(X_df_chunk, sequence_len=SEQUENCE_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference on this chunk
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}", leave=False):
            writer = make_batch_embeddings(
                device, model, OUTPUT_PATH, writer, cols, batch
            )

    # Clean up RAM immediately
    del X_df_chunk, dataset, dataloader, batch_chunk
    gc.collect()  # Force Python to free memory


def make_batch_embeddings(device, model, OUTPUT_PATH, writer, cols, batch):
    """
    Generates embeddings for a single batch of data and writes them to a Parquet file.

    Args:
        device (torch.device): The device (CPU or CUDA) to run the model on.
        model (PatchedTimeSeriesEmbeddingGenerator): The FinCast embedding model.
        OUTPUT_PATH (str): The path to the output Parquet file.
        writer (pq.ParquetWriter): The Parquet writer object.
        cols (list): A list of column names for the embeddings.
        batch (dict): A dictionary containing the batch values and indices.

    Returns:
        pq.ParquetWriter: The updated Parquet writer object.
    """
    batch_vals = batch["vals"].to(device, non_blocking=True)
    batch_mask = batch["mask"].to(device, non_blocking=True)
    batch_indices = batch["index"]

    batch_freq = torch.zeros(batch_vals.shape[0], dtype=torch.long, device=device)

    if torch.cuda.is_available():
        # USE float16 for T4 compatibility
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            embeddings = model(batch_vals, batch_mask, batch_freq)
    else:
        embeddings = model(batch_vals, batch_mask, batch_freq)

    aggregated_embeddings = embeddings.mean(dim=1).cpu()

    final_array = aggregated_embeddings.numpy()

    # Create column names on the first pass
    if cols is None:
        cols = [f"emb_{j}" for j in range(final_array.shape[1])]

    if isinstance(batch_indices, torch.Tensor):
        batch_indices = batch_indices.tolist()

    emb_df = pd.DataFrame(final_array, columns=cols, index=batch_indices)

    table = pa.Table.from_pandas(emb_df)

    # Initialize writer on the first pass
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="zstd")

    writer.write_table(table=table)
    return writer


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # Default model configuration
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

    # Use GPU if it is avaiable
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found.")
        device = torch.device("cpu")

    embedder = PatchedTimeSeriesEmbeddingGenerator(config)

    # Load the weights for the model
    print("Loading weights...")
    state_dict = torch.load("weights/v1.pth", map_location="cpu", weights_only=True)
    embedder.load_state_dict(state_dict, strict=True)
    embedder.to(device)
    embedder.eval()

    make_embeddings("ptb-xl", batch_size=32, device=device, model=embedder)
    make_embeddings("esc-50", batch_size=2, device=device, model=embedder)
    make_embeddings("SP-500", batch_size=256, device=device, model=embedder)
