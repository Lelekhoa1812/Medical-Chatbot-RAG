from pymongo import MongoClient
from dotenv import load_dotenv
import os

# # Load environment variables from .env
# load_dotenv()

##-------------##
# FOR QA CLUSTER
##-------------##

# mongo_uri = os.getenv("MONGO_URI")
# if not mongo_uri:
#     raise ValueError("❌ MongoDB URI (MongoURI) is missing!")

# client = MongoClient(mongo_uri)
# db = client["MedicalChatbotDB"]  # Use the same database name as in your main script

# # To drop just the collection storing the FAISS index:
# db.drop_collection("qa_data")
# print("Dropped collection 'qa_data' from MedicalChatbotDB.")

# # Alternatively, to drop the entire database:
# client.drop_database("MedicalChatbotDB")
# print("Dropped database 'MedicalChatbotDB'.")


##-------------##
# FOR INDEX CLUSTER
##-------------##

# Load environment variables from .env
# load_dotenv()
# index_uri = os.getenv("INDEX_URI")
# if not index_uri:
#     raise ValueError("❌ MongoDB URI (IndexURI) is missing!")

# iclient = MongoClient(index_uri)
# idb = iclient["MedicalChatbotDB"]  # Use the same database name as in your main script

# # To drop just the collection storing the FAISS index:
# idb.drop_collection("faiss_index_files.files")
# idb.drop_collection("faiss_index_files.chunks")
# print("Dropped collection 'faiss_index_files' and chunks from MedicalChatbotDB.")

# # Alternatively, to drop the entire database:
# iclient.drop_database("MedicalChatbotDB")
# print("Dropped database 'MedicalChatbotDB'.")