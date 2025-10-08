# Running this script to split FAISS index collection to the second/different cluster.
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def migrate_faiss_index():
    """Migrate FAISS index from QA cluster to index cluster"""
    # Load environment variables from .env
    load_dotenv()
    # Connection strings (update as needed)
    mongo_uri = os.getenv("MONGO_URI")  # QA cluster connection string
    index_uri = os.getenv("INDEX_URI")  # FAISS index cluster connection string

    if not mongo_uri:
        raise ValueError("MONGO_URI is missing!")
    if not index_uri:
        raise ValueError("INDEX_URI is missing!")

    # Connect to the QA cluster (where FAISS data was accidentally stored)
    qa_client = MongoClient(mongo_uri)
    qa_db = qa_client["MedicalChatbotDB"]

    # Connect to the FAISS index cluster
    faiss_client = MongoClient(index_uri)
    faiss_db = faiss_client["MedicalChatbotDB"]  # Use the same database name if desired

    # Define the GridFS collections to move.
    # In GridFS, files are stored in two collections: "<bucket>.files" and "<bucket>.chunks".
    source_files = qa_db["faiss_index_files.files"]
    source_chunks = qa_db["faiss_index_files.chunks"]

    dest_files = faiss_db["faiss_index_files.files"]
    dest_chunks = faiss_db["faiss_index_files.chunks"]

    print("Moving FAISS index GridFS files...")

    # Copy documents from the source 'files' collection
    for doc in source_files.find():
        dest_files.insert_one(doc)

    # Copy documents from the source 'chunks' collection
    for doc in source_chunks.find():
        dest_chunks.insert_one(doc)

    print("✅ FAISS GridFS collections moved successfully.")

    # Optionally, drop the old collections from the QA cluster to free up space:
    qa_db.drop_collection("faiss_index_files.files")
    qa_db.drop_collection("faiss_index_files.chunks")
    print("Old FAISS GridFS collections dropped from the QA cluster.")

# Only run when called directly
if __name__ == "__main__":
    migrate_faiss_index()
