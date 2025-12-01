# Purpose:
# Makes wave2vec embeddings for S&P 500, ESC 50, PTB XL Datasets.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py

# Authors:
# - Code written by Philip Loewen

# Reference:
# A. Baevski, H. Zhou, M. Abdelrahman, M. Auli, "wav2vec 2.0: a framework for
# self-supervised learning of speech representations," Neural Information Processing
# Systems, vol. 33, pp. 12449–12460, Jun. 2020, [Online].
# Available: https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf

from transformers import AutoProcessor, Wav2Vec2Model
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)


def make_data_embeddings(data):
    for split in ["train", "test"]:
        print(f"\nProcessing {data}-{split} on {device}...")
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/W2V-{data}-{split}.parquet"

        X_df = pd.read_parquet(INPUT_PATH)
        X_np = X_df["target"]

        n = X_df.shape[0]

        all_pooled_embeddings = []

        for i in tqdm(range(0, n), desc="Extracting"):
            X = X_np[i]

            m = len(X)
            if m < 400:
                pad = np.zeros(400 - m)
                batch = np.hstack([X, pad])
            else:
                batch = X

            inputs = processor(batch, sampling_rate=16000, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                outputs = model(**inputs)

            pooled_embedding = outputs["last_hidden_state"].mean(dim=1).cpu()
            all_pooled_embeddings.append(pooled_embedding)

        final_tensor = torch.cat(all_pooled_embeddings, dim=0)

        concatenated_tensor = final_tensor.reshape(-1, 2 * final_tensor.shape[-1])

        new_index = X_df.index[::2][: concatenated_tensor.shape[0]]

        final_array = concatenated_tensor.cpu().numpy()

        cols = [f"emb_{i}" for i in range(final_array.shape[1])]
        emb_df = pd.DataFrame(final_array, columns=cols, index=new_index)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
        print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    make_data_embeddings("star")
