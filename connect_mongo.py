from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Test MongoDB connection, and list out all collection.
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["MedicalChatbotDB"]
print(db.list_collection_names())
