from transformers import AutoProcessor, Wav2Vec2Model
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import math

# --- MacOS Optimizations ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# reduced thread count to prevent fighting for resources if using CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Check for Apple Silicon (MPS) or CUDA, fallback to CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple MPS (Metal Performance Shaders) acceleration.")
else:
    device = "cpu"
    print("Using CPU.")

print(f"Loading model to {device}...")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()  # Set model to evaluation mode


def make_data_embeddings(data, batch_size=32):
    for split in ["train", "test"]:
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/W2V-{data}-{split}.parquet"

        if not os.path.exists(INPUT_PATH):
            print(f"Skipping {INPUT_PATH} (file not found)")
            continue

        print(f"\nProcessing {data}-{split}...")
        X_df = pd.read_parquet(INPUT_PATH)
        X_np = X_df["target"].values

        n = len(X_np)
        all_pooled_embeddings = []

        # --- Batch Processing Loop ---
        # Processes 'batch_size' items at a time instead of 1
        for i in tqdm(range(0, n, batch_size), desc=f"Extracting {split}"):
            batch_raw = X_np[i : i + batch_size]

            # Pad sequences shorter than 400 manually (vectorized)
            # Note: Processor can also handle padding, but keeping your logic for consistency
            processed_batch = []
            for x in batch_raw:
                m = len(x)
                if m < 400:
                    pad = np.zeros(400 - m)
                    processed_batch.append(np.hstack([x, pad]))
                else:
                    processed_batch.append(x)

            # Tokenize/Process batch
            inputs = processor(
                processed_batch, sampling_rate=16000, return_tensors="pt", padding=True
            ).to(device)

            with torch.inference_mode():  # More efficient than no_grad
                outputs = model(**inputs)

            # Mean pooling over the time dimension
            pooled_embedding = outputs.last_hidden_state.mean(dim=1).cpu()
            all_pooled_embeddings.append(pooled_embedding)

        # Concatenate all batches
        final_tensor = torch.cat(all_pooled_embeddings, dim=0)

        # --- Safe Reshape Logic ---
        # Your logic combines every 2 rows into 1 row (doubling width).
        # We must ensure we have an even number of rows.
        if final_tensor.shape[0] % 2 != 0:
            print(
                f"Warning: {split} set has odd number of rows ({final_tensor.shape[0]}). Dropping last row to allow pairing."
            )
            final_tensor = final_tensor[:-1]
            # Adjust index to match dropped row
            valid_index_limit = 2 * (final_tensor.shape[0] // 2)
            current_index = X_df.index[:valid_index_limit]
        else:
            current_index = X_df.index

        # Reshape: (N, D) -> (N/2, D*2)
        concatenated_tensor = final_tensor.reshape(-1, 2 * final_tensor.shape[-1])

        # Adjust index: take every 2nd index
        new_index = current_index[::2]

        final_array = concatenated_tensor.numpy()
        cols = [f"emb_{i}" for i in range(final_array.shape[1])]

        emb_df = pd.DataFrame(final_array, columns=cols, index=new_index)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
        print(f"Saved {emb_df.shape} to {OUTPUT_PATH}")


if __name__ == "__main__":
    # Ensure raw audio loading works with multiprocessing on macOS
    mp_start_method = "fork" if hasattr(os, "fork") else "spawn"
    try:
        import multiprocessing as mp

        mp.set_start_method(mp_start_method, force=True)
    except RuntimeError:
        pass

    make_data_embeddings("star", batch_size=64)
