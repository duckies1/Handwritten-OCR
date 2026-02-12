import os
from huggingface_hub import snapshot_download

# ==========================
# CONFIG
# ==========================
MODEL_REPO = "Duckies1/Handwritten-OCR"
DATASET_REPO = "Duckies1/IAM-line"

MODEL_DIR = "model"
DATASET_DIR = "dataset"

# ==========================
# CREATE DIRECTORIES
# ==========================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

print("Downloading model from Hugging Face...")
snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False
)

print("Downloading dataset from Hugging Face...")
snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    local_dir=DATASET_DIR,
    local_dir_use_symlinks=False
)

print("\n✅ Download complete!")
print(f"Model saved to: {MODEL_DIR}/")
print(f"Dataset saved to: {DATASET_DIR}/")
