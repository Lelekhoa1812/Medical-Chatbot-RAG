# download_model.py
### --- A. transformer and embedder ---
import os
import shutil
from huggingface_hub import snapshot_download

# Set up paths
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = "/app/model_cache"

print("⏳ Downloading the SentenceTransformer model...")
model_path = snapshot_download(repo_id=MODEL_REPO, cache_dir=MODEL_CACHE_DIR)

print("Model path: ", model_path)

# Ensure the directory exists
if not os.path.exists(MODEL_CACHE_DIR):
    os.makedirs(MODEL_CACHE_DIR)

# Move all contents from the snapshot folder
if os.path.exists(model_path):
    print(f"📂 Moving model files from {model_path} to {MODEL_CACHE_DIR}...")

    for item in os.listdir(model_path):
        source = os.path.join(model_path, item)
        destination = os.path.join(MODEL_CACHE_DIR, item)

        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    print(f"✅ Model extracted and flattened in {MODEL_CACHE_DIR}")
else:
    print("❌ No snapshot directory found!")
    exit(1)

# Verify structure after moving
print("\n📂 LLM Model Structure (Build Level):")
for root, dirs, files in os.walk(MODEL_CACHE_DIR):
    print(f"📁 {root}/")
    for file in files:
        print(f"  📄 {file}")


### --- B. translation modules ---
from transformers import pipeline
print("⏬ Downloading Vietnamese–English translator...")
_ = pipeline("translation", model="VietAI/envit5-translation", src_lang="vi", tgt_lang="en")
print("⏬ Downloading Chinese–English translator...")
_ = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")