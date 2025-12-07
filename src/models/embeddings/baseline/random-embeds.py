# Purpose:
# Makes random embeddings for S&P 500, ESC 50, PTB XL Datasets.
#
# Embeddings are made using iid Uniform(0, 1) random variables.

# Pre-requisites:
# - Requires running process-SP-500.py, process-esc-50.py, process-ptb-xl.py

# Authors:
# - Code written by Philip Loewen

import numpy as np
import pandas as pd
import os
from pyarrow.parquet import ParquetFile


def get_embeddings(data):
    for split in ["test", "train"]:
        print(f"Processing {data}-{split}...")
        input_path = f"data/processed/{data}/{data}-X-{split}.parquet"
        output_path = f"data/embeddings/{data}/RANDOM-{data}-{split}.parquet"

        df = ParquetFile(input_path)
        nrows = df.metadata.num_rows

        random_array = np.random.rand(nrows, 512)
        cols = [f"emb_{i}" for i in range(random_array.shape[1])]

        emb_df = pd.DataFrame(random_array, columns=cols)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        emb_df.to_parquet(output_path, compression="zstd")

        print(f"Saved {output_path}\n")


if __name__ == "__main__":
    get_embeddings("star")
    get_embeddings("esc-50")
    get_embeddings("ptb-xl")
    get_embeddings("SP-500")
