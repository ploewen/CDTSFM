import pandas as pd
import numpy as np
import torch
import os
from ASTROMER.models import SingleBandEncoder


def make_data_embeddings(data, model, batch_size):
    for split in ["test", "train"]:
        print(f"\nProcessing {data}-{split}...")
        INPUT_PATH = f"data/processed/{data}/{data}-X-{split}.parquet"
        OUTPUT_PATH = f"data/embeddings/{data}/ASTRO-{data}-{split}.parquet"

        X_df = pd.read_parquet(INPUT_PATH)

        print("Reshaping dataframe...")

        all_augm_df, lengths = reshape_data(X_df)

        print("Making embeddings...")
        num_samples = len(all_augm_df)
        # labels = np.zeros(num_samples)
        # oids = np.arange(num_samples)
        # embeds = model.encode(
        #     all_augm_df, labels=labels, oids_list=oids, batch_size=batch_size
        # )
        embeds = model.encode(all_augm_df, batch_size=batch_size)

        print("Aggregating embeddings...")

        final_array = aggregate_embeddings(lengths, embeds)

        print(f"Creating DataFrame with shape {final_array.shape}...")
        # Create Column Names
        cols = [f"emb_{i}" for i in range(final_array.shape[1])]
        emb_df = pd.DataFrame(final_array, columns=cols, index=X_df.index)

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
        print(f"Successfully saved to {OUTPUT_PATH}")


def aggregate_embeddings(lengths, embeds):
    embed_tensor = torch.tensor(embeds)
    pooled_embeds = embed_tensor.mean(dim=1)

    final_pooled_embeds = []
    if not all(x == 1 for x in lengths):
        k = 0
        for length in lengths:
            sample_embed = pooled_embeds[k : k + length].mean(dim=0)
            k += length
            final_pooled_embeds.append(sample_embed)
        torch.tensor(final_pooled_embeds)
    else:
        final_pooled_embeds = pooled_embeds

    final_array = final_pooled_embeds.numpy()
    return final_array


def reshape_data(X_df):
    n = X_df.shape[0]
    m = X_df.shape[1]

    all_augm_df = []
    lengths = [0] * n
    for i in range(n):
        if m > 200:
            for k in range(0, m, 200):
                augm_df = np.zeros((200, 3))
                augm_df[:, 0] = X_df.iloc[i, k : k + 200]
                augm_df[:, 1] = np.arange(200)
                all_augm_df.append(augm_df)
                lengths[i] += 1
        else:
            augm_df = np.zeros((m, 3))
            augm_df[:, 0] = X_df.iloc[i]
            augm_df[:, 1] = np.arange(m)
            all_augm_df.append(augm_df)
            lengths[i] = 1
    return all_augm_df, lengths


if __name__ == "__main__":
    model = SingleBandEncoder()
    model = model.from_pretraining("macho")
    make_data_embeddings("SP-500", model, 32)
    make_data_embeddings("esc-50", model, 1)
    make_data_embeddings("ptb-xl", model, 16)
