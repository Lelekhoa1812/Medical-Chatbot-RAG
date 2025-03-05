import os
import shutil
from huggingface_hub import snapshot_download

# Define the target cache directory
MODEL_CACHE_DIR = "/app/model_cache"

# Download model
print("⏳ Downloading the SentenceTransformer model...")
model_path = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", cache_dir=MODEL_CACHE_DIR)

# Find the snapshot folder
snapshots_dir = os.path.join(model_path, "snapshots")
if os.path.exists(snapshots_dir):
    snapshot_subdirs = os.listdir(snapshots_dir)
    if snapshot_subdirs:
        snapshot_dir = os.path.join(snapshots_dir, snapshot_subdirs[0])
        # Move all files to the main model cache directory
        for filename in os.listdir(snapshot_dir):
            shutil.move(os.path.join(snapshot_dir, filename), MODEL_CACHE_DIR)
print(f"✅ Model downloaded and stored in {MODEL_CACHE_DIR}")
