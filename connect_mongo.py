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
print("QA Collection: ",db.list_collection_names())
# Count document QA related
print("QA count: ", db.qa_data.count_documents({}))

# Index Cluster
index_uri = os.getenv("INDEX_URI")
iclient = MongoClient(index_uri)
idb = iclient["FAISSIndexCluster"]
# List all collection
print("FAISS Collection: ",idb.list_collection_names())
# Count document QA related
print("Index count: ", idb.faiss_index.count_documents({}))