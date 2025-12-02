import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import gc
import numpy as np

# Import your model components
from ffm.pytorch_patched_decoder_MOE import (
    PatchedTimeSeriesDecoder_MOE,
    FFMConfig,
    create_quantiles,
)

# --- CONFIGURATION ---
NUM_WORKERS = 4  # Increased for data loading speed
CHUNK_SIZE = 5000
BATCH_SIZE = 128
SEQUENCE_LENGTH = 32


class PatchedTimeSeriesEmbeddingGenerator(PatchedTimeSeriesDecoder_MOE):
    """
    A time-series embedding generator adapted from the PatchedTimeSeriesDecoder_MOE class.
    Modifies forward pass to return embeddings before the prediction head.
    """

    def __init__(self, config: FFMConfig):
        super().__init__(config)

    def forward(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
    ) -> torch.Tensor:
        model_input, patched_padding, _, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )

        f_emb = self.freq_emb(freq).unsqueeze(1)
        model_input += f_emb

        model_output, _ = self.stacked_transformer(model_input, patched_padding)

        return model_output


class RaggedDataset(Dataset):
    def __init__(self, df):
        self.data = df.iloc[:, 0].values
        self.indices = df.index.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "vals": torch.tensor(self.data[idx], dtype=torch.float32),
            "index": self.indices[idx],
        }


def pad_sequence(batch):
    sequences = [item["vals"] for item in batch]
    indices = [item["index"] for item in batch]

    max_len = max(s.size(0) for s in sequences)

    remainder = max_len % SEQUENCE_LENGTH
    if remainder != 0:
        target_len = max_len + (SEQUENCE_LENGTH - remainder)
    else:
        target_len = max_len

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

    return {"vals": torch.stack(padded_batch), "index": indices}


def make_embeddings(data, batch_size, device, model):
    for split in ["test", "train"]:
        try:
            output_path, input_path = get_data(data, split)
        except FileNotFoundError:
            print(f"File not found: {data}-{split}, skipping...")
            continue

        # Read entire parquet file into a Pandas DataFrame
        print(f"Reading {input_path}...")
        X_df_full = pd.read_parquet(input_path)

        make_all_embeddings(
            X_df_full,
            batch_size,
            model,
            output_path,
            device,
        )

        print(f"Successfully saved to {output_path}")
        gc.collect()


def get_data(data, split):
    print(f"\nProcessing {data}-{split}...")
    input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
    output_path = f"data/embeddings/{data}/FinCast-{data}-{split}.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError

    return output_path, input_path


def make_all_embeddings(X_df_full, batch_size, model, OUTPUT_PATH, device):
    dataset = RaggedDataset(X_df_full)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_sequence,
    )

    writer = None
    schema = None  # Used to lock the schema after the first batch

    # Counter for GC
    batch_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing embeddings", leave=False):
            writer, schema = make_batch_embeddings(
                device, model, OUTPUT_PATH, writer, schema, batch
            )

            batch_counter += 1
            if batch_counter % 10 == 0:
                gc.collect()

    if writer:
        writer.close()


def make_batch_embeddings(device, model, OUTPUT_PATH, writer, schema, batch):
    """
    Generates and writes embeddings, ensuring Schema consistency across batches.
    """
    batch_vals = batch["vals"].to(device, dtype=torch.float32, non_blocking=True)
    batch_mask = (batch_vals != 0).type_as(batch_vals)
    batch_indices = batch["index"]
    batch_freq = torch.zeros(batch_vals.shape[0], dtype=torch.long, device=device)

    # Inference
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            embeddings = model(batch_vals, batch_mask, batch_freq)
    else:
        embeddings = model(batch_vals, batch_mask, batch_freq)

    # Aggregate: Mean over time dimension
    # CRITICAL FIX: Force to float32 immediately to prevent "halffloat" schema issues
    aggregated_embeddings = embeddings.mean(dim=1).float().cpu()
    final_array = aggregated_embeddings.numpy()

    # --- ROW CONCATENATION LOGIC ---
    # Concatenate every two rows (StarEmbed logic)
    if final_array.shape[0] % 2 != 0:
        # Pad with zeros if odd number of rows
        final_array = np.vstack(
            [final_array, np.zeros((1, final_array.shape[1]), dtype=np.float32)]
        )
        batch_indices.append(batch_indices[-1])  # Duplicate last index

    # Reshape: (Batch/2, Hidden*2)
    concatenated_array = final_array.reshape(final_array.shape[0] // 2, -1)
    concatenated_indices = [batch_indices[i] for i in range(0, len(batch_indices), 2)]

    # Generate column names if not done yet
    cols = [f"emb_{j}" for j in range(concatenated_array.shape[1])]

    # Create DataFrame
    emb_df = pd.DataFrame(concatenated_array, columns=cols, index=concatenated_indices)

    # Convert to PyArrow Table
    if schema is not None:
        # For batch 2+, enforce the schema from batch 1
        # This fixes the "vs file" error by ensuring metadata/types match exactly
        table = pa.Table.from_pandas(emb_df, schema=schema)
    else:
        # For batch 1, infer schema
        table = pa.Table.from_pandas(emb_df)
        schema = table.schema

    # Initialize writer if first batch
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, schema, compression="zstd")

    writer.write_table(table=table)

    return writer, schema


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
    # Ensure map_location handles the device correctly
    state_dict = torch.load("weights/v1.pth", map_location="cpu", weights_only=True)
    embedder.load_state_dict(state_dict, strict=True)
    embedder.to(device)
    embedder.eval()

    # Run the main process
    make_embeddings("star", batch_size=BATCH_SIZE, device=device, model=embedder)
