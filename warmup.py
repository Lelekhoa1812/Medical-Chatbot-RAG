from sentence_transformers import SentenceTransformer
import torch

print("🚀 Warming up model...")
embedding_model = SentenceTransformer("/app/model_cache", device="cpu")
embedding_model = embedding_model.half()  # Reduce memory
embedding_model.to(torch.device("cpu"))
print("✅ Model warm-up complete!")
