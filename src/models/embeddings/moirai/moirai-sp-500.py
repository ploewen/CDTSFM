import pandas as pd
import torch
import os
from uni2ts.model.moirai2 import Moirai2Module
from tqdm import tqdm

# Set the device to use GPU of possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def make_moirai_embeddings(SPLIT):
    # CONFIG
    BATCH_SIZE = 32
    PATCH_SIZE = 16  # Fixed by Model
    SEQ_LEN = 80  # 16 * 5 Patches
    INPUT_PATH = "data/processed/SP-500/SP-500-X-" + SPLIT + ".parquet"
    OUTPUT_DIR = "data/embeddings/SP-500/"
    FILE_NAME = "MOIRAI-SP-500-" + SPLIT + ".parquet"
    OUTPUT_PATH = OUTPUT_DIR + FILE_NAME

    print(f"Making {split} embeddings.")

    # Make the output directory if it does not yet exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load data
    print(f"Loading data from {INPUT_PATH}...")
    X_train = pd.read_parquet(INPUT_PATH)
    X_arr = X_train.values

    # Check that we have the right number of columns
    if X_arr.shape[1] != SEQ_LEN:
        raise ValueError(
            f"Input data has {X_arr.shape[1]} columns, but expected {SEQ_LEN}."
        )

    # Set model to moirai 2 small
    model = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    model.to(device)
    model.eval()

    # Hook storage
    embeddings_storage = {}

    def get_embeddings(name):
        def hook(model, input, output):
            embeddings_storage[name] = output.detach().cpu()

        return hook

    model.encoder.register_forward_hook(get_embeddings("backbone_output"))

    # Get all the embeddings into a list
    all_pooled_embeddings = []

    # Calculate number of patches for reshaping
    num_patches = SEQ_LEN // PATCH_SIZE

    print("Making embeddings...")
    # Loop through data in chunks
    for i in tqdm(range(0, len(X_arr), BATCH_SIZE)):
        # Prepare batch
        batch_raw = X_arr[i : i + BATCH_SIZE]
        current_bs = len(batch_raw)

        # Reshape the batch: Input: (Batch, 80) -> Output: (Batch, 5, 16)
        target_patched = torch.tensor(batch_raw, dtype=torch.float32).reshape(
            current_bs, num_patches, PATCH_SIZE
        )

        # Create Masks & IDs
        # Since data is full length (no padding), observed_mask is all True
        observed_mask = torch.ones(
            (current_bs, num_patches, PATCH_SIZE), dtype=torch.bool
        )

        # Time ID: [0, 1, 2, 3, 4] repeated for batch
        time_id = torch.arange(num_patches, dtype=torch.int64).expand(current_bs, -1)

        # Zeros for others
        variate_id = torch.zeros((current_bs, num_patches), dtype=torch.int64)
        sample_id = torch.zeros((current_bs, num_patches), dtype=torch.int64)
        prediction_mask = torch.zeros((current_bs, num_patches), dtype=torch.bool)

        # Move to Device
        inputs = {
            "target": target_patched.to(device),
            "observed_mask": observed_mask.to(device),
            "sample_id": sample_id.to(device),
            "time_id": time_id.to(device),
            "variate_id": variate_id.to(device),
            "prediction_mask": prediction_mask.to(device),
            "training_mode": False,
        }

        # Pass our inputs into the model
        with torch.no_grad():
            model(**inputs)

        # Extract embeddings
        raw_emb = embeddings_storage["backbone_output"]

        # Go From [Batch, 5, 384] -> [Batch, 384]
        pooled_emb = raw_emb.mean(dim=1)

        # Add pooled embedding to list
        all_pooled_embeddings.append(pooled_emb)

    # Save the results
    print("Concatenating results...")
    final_tensor = torch.cat(all_pooled_embeddings, dim=0)
    final_array = final_tensor.numpy()

    print(f"Creating DataFrame with shape {final_array.shape}...")
    # Create Column Names
    cols = [f"emb_{i}" for i in range(final_array.shape[1])]
    emb_df = pd.DataFrame(final_array, columns=cols, index=X_train.index)

    emb_df.to_parquet(OUTPUT_PATH, compression="zstd")
    print(f"Successfully saved to {OUTPUT_PATH}")


for split in ["train", "test"]:
    make_moirai_embeddings(split)
