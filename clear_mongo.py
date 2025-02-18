from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
mongo_uri = os.getenv("MongoURI")
if not mongo_uri:
    raise ValueError("❌ MongoDB URI (MongoURI) is missing!")

client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]  # Use the same database name as in your main script

# To drop just the collection storing the FAISS index:
db.drop_collection("qa_data")
print("Dropped collection 'faiss_index' from MedicalChatbotDB.")

# Alternatively, to drop the entire database:
# client.drop_database("MedicalChatbotDB")
# print("Dropped database 'MedicalChatbotDB'.")