from transformers import AutoProcessor, Wav2Vec2Model
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)


def make_data_embeddings(data, batch_size=8):
    for split in ["train", "test"]:
        print(f"\nProcessing {data}-{split} on {device}...")
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/W2V-{data}-{split}.parquet"

        X_df = pd.read_parquet(INPUT_PATH)
        X_np = X_df.values

        n = X_df.shape[0]
        m = X_df.shape[1]

        if m < 400:
            pad = np.zeros((n, 400 - m))
            X_padded = np.hstack([X_np, pad])
        else:
            X_padded = X_np
        all_pooled_embeddings = []

        for i in tqdm(range(0, n, batch_size), desc="Extracting"):
            batch = X_padded[i : i + batch_size]

            inputs = processor(batch, sampling_rate=16000, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                outputs = model(**inputs)

            pooled_embedding = outputs["last_hidden_state"].mean(dim=1).cpu()
            all_pooled_embeddings.append(pooled_embedding)

        final_tensor = torch.cat(all_pooled_embeddings, dim=0)
        final_array = final_tensor.numpy()

        cols = [f"emb_{i}" for i in range(final_array.shape[1])]
        emb_df = pd.DataFrame(final_array, columns=cols, index=X_df.index)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
        print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    # make_data_embeddings("ESC-50", 8)
    make_data_embeddings("ptb-xl", 8)
    make_data_embeddings("SP-500", 64)
