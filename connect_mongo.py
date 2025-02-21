from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Test MongoDB connection, and list out all collection.
load_dotenv()

# QA Cluster
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]
# List all collection
print("Collection: ",db.list_collection_names())
# Count document QA related
print("QA count: ", db.qa_data.count_documents({}))

# Index Cluster
index_uri = os.getenv("INDEX_URI")
iclient = MongoClient(index_uri)
index_db = iclient["MedicalChatbotDB"]
# List all collection
print("Collection: ",index_db.list_collection_names())
# Count document QA related
print("Index count: ", index_db.faiss_index.count_documents({}))