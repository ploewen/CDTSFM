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

        random_embed = pd.DataFrame(np.random.rand(nrows, 512))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

        random_embed.to_parquet(output_path)

        print(f"Saved {output_path}\n")


if __name__ == "__main__":
    get_embeddings("star")
    get_embeddings("esc-50")
    get_embeddings("ptb-xl")
    get_embeddings("SP-500")
