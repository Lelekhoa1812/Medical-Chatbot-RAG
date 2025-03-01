### **🔍 Estimated Total Storage on Render**
| Component | Approximate Size |
|-----------|----------------|
| Sentence Transformer Model | **~90MB** |
| FAISS Index (stored externally) | **0MB** |
| Python Dependencies | **~150MB-200MB** |
| Render OS Files | **~150MB** |
| **Total (on Render)** | **~400MB-450MB** |

- ✅ **LIKELY within the 500MB limit!**
- ⚠️ **BUT**: If you later add **larger models** or **more dependencies**, it might exceed **500MB**.

---

### **🔥 How to Reduce Storage Usage?**
**✅ 1. Reduce Python Dependencies**

**✅ 2. Use Hugging Face Model Cache Externally**
- Instead of downloading the model locally, cache it externally:
  ```sh
  os.environ["HF_HOME"] = "/dev/shm/huggingface_models"  # Temporary cache in RAM
  ```

**✅ 3. Optimize FAISS Index**
- Store **only required dimensions** and use `IVFPQ` compression instead of `HNSWFlat`:
  ```python
  quantizer = faiss.IndexFlatL2(384)  # Use only required 384 dimensions
  index = faiss.IndexIVFPQ(quantizer, 384, 64, 8, 8)  # Compressed FAISS index
  ```

**✅ 4. Consider Render's Paid Plan**
- If the project **eventually exceeds** 500MB, consider Render’s **paid plan ($7/month)**.

---
