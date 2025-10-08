# ✅ Google Colab: SymbiPredict Embedding + Chunking + MongoDB Upload

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import hashlib, os
from tqdm import tqdm

# ✅ Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load SymbiPredict
df = pd.read_csv("symbipredict_2022.csv")

# ✅ Connect to MongoDB
mongo_uri = "..."
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]
collection = db["symptom_diagnosis"]

# ✅ Clear old symptom-diagnosis records
print("🧹 Dropping old 'symptom_diagnosis' collection...")
collection.drop()
#  Reconfirm collection is empty
if collection.count_documents({}) != 0:
    raise RuntimeError("❌ Collection not empty after drop — aborting!")

# ✅ Convert CSV rows into QA-style records with embeddings
records = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    symptom_cols = df.columns[:-1]
    label_col = df.columns[-1]

    # Extract symptoms present (value==1)
    symptoms = [col.replace("_", " ").strip() for col in symptom_cols if row[col] == 1]
    if not symptoms:
        continue

    label = row[label_col].strip()
    question = f"What disease is likely given these symptoms: {', '.join(symptoms)}?"
    answer = f"The patient is likely suffering from: {label}."

    # Embed question only
    embed = model.encode(question, convert_to_numpy=True)
    hashkey = hashlib.md5((question + answer).encode()).hexdigest()

    records.append({
        "_id": hashkey,
        "i": int(i),
        "symptoms": symptoms,
        "prognosis": label,
        "question": question,
        "answer": answer,
        "embedding": embed.tolist()
    })

# ✅ Save to MongoDB
if records:
    print(f"⬆️ Uploading {len(records)} records to MongoDB...")
    unique_ids = set()
    deduped = []
    for r in records:
        if r["_id"] not in unique_ids:
            unique_ids.add(r["_id"])
            deduped.append(r)
    try:
      collection.insert_many(deduped, ordered=False)
      print(f"✅ Inserted {len(deduped)} records without duplicates.")
    except BulkWriteError as bwe:
      inserted = bwe.details.get('nInserted', 0)
      print(f"⚠️ Inserted with some duplicate skips. Records inserted: {inserted}")
    print("✅ Upload complete.")
else:
    print("⚠️ No records to upload.")