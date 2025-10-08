from pymongo import MongoClient
from dotenv import load_dotenv
import os

def test_mongodb_connections():
    """Test MongoDB connections and list collections"""
    load_dotenv()

    # QA Cluster
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("❌ MONGO_URI environment variable not set!")
        return
        
    client = MongoClient(mongo_uri)
    db = client["MedicalChatbotDB"]
    # List all collection
    print("QA Collection: ", db.list_collection_names())
    # Count document QA related
    print("QA count: ", db.qa_data.count_documents({}))

    # Index Cluster
    index_uri = os.getenv("INDEX_URI")
    if not index_uri:
        print("❌ INDEX_URI environment variable not set!")
        return
        
    iclient = MongoClient(index_uri)
    idb = iclient["MedicalChatbotDB"]
    # List all collection
    print("FAISS Collection: ", idb.list_collection_names())
    # Count document QA related
    print("Index count: ", idb.faiss_index_files.files.count_documents({}))

# Only run when called directly
if __name__ == "__main__":
    test_mongodb_connections()