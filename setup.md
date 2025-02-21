## **AutoGen-Based Medical Chatbot with RAG**

This guide will help you deploy a **Medical Chatbot** using **AutoGen**, **RAG (Retrieval-Augmented Generation)**, and **OpenAI API** with a **custom Hugging Face medical dataset**.

---

## **Installation**
### **Step 1: Set Up Conda Environment**
Create a new Conda environment with Python 3.11:   
```bash
conda create -n chatbot_env python=3.11 -y
conda activate chatbot_env
```
**Notice:** Ensure your machine has required Anaconda or Miniconda version installed.  

---

### **Step 2: Install Dependencies**
Ensure your environment has the necessary libraries:   
```bash
pip install -U "autogen-agentchat" "autogenstudio" \
                "faiss-cpu" "chromadb" "datasets" "tiktoken" "sentence-transformers" "python-dotenv" "google-genai" "huggingface_hub" "pymongo"
```

- `faiss-cpu`: Efficient vector search for RAG
- `chromadb`: Local vector database for retrieval
- `datasets`: Loading and processing custom data
- `sentence-transformers`: Converting text into embeddings
- `python-dotenv`: Managing OpenAI API keys securely
- `pymongo`: Utilization of MongoDB and gridfs (for compressed FAISS data saving)

Installation of server and router on FastAPI for a client-server interface:  
```bash
pip install fastapi uvicorn requests python-multipart
```

---

## **Configuration**
### **Step 3: Set Up OpenAI API Key**
To integrate RAG and answer intelligently, you'll need an **OpenAI API key**:  
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Alternatively, store it in a `.env` file:
```bash
echo 'OPENAI_API_KEY="your_openai_api_key"' > .env
```
### **Optional: Saving embedded vector database to MongoDB**
Store MongoDB cluster API key into `.env` file:
```bash
echo 'MongoURI=your_mongodb_api_key' > .env
```
---

## **Dataset Preparation**
### **Step 4: Load the Medical Dataset from Hugging Face**

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot")

# Extract patient-doctor conversations
medical_dialogues = dataset["train"].to_pandas()[["Patient", "Doctor"]]
print("Example:", medical_dialogues.iloc[0])
```

---

## **Build FAISS Index for RAG**
### **Step 5: Convert Dataset into Embeddings**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into embeddings
medical_qa = [
    {"question": row["Patient"], "answer": row["Doctor"]}
    for _, row in medical_dialogues.iterrows()
]

# Generate vector embeddings
medical_embeddings = embedding_model.encode(
    [qa["question"] + " " + qa["answer"] for qa in medical_qa],
    convert_to_numpy=True
)

# Create FAISS index
index = faiss.IndexFlatL2(medical_embeddings.shape[1])
index.add(medical_embeddings)
faiss.write_index(index, "data/medical_faiss_index")
print("FAISS index saved successfully!")
```

---

## **AutoGen Chatbot Implementation**
### **Step 6: Create the Chatbot with RAG**

```python
import autogen_agentchat
import autogen_ext 
import faiss

# Load FAISS index
index = faiss.read_index("data/medical_faiss_index")

def retrieve_medical_info(query):
    """Retrieve relevant medical knowledge using FAISS"""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    _, idxs = index.search(query_embedding, k=3)  # Get top 3 matches
    return [medical_qa[i]["answer"] for i in idxs[0]]

class MedicalChatbot(autogen_agentchat.AssistantAgent):  # Use autogen_agentchat
    def generate_reply(self, messages):
        query = messages[-1]["content"]  # Get latest user query
        retrieved_info = retrieve_medical_info(query)
        knowledge_base = "\n".join(retrieved_info)

        # Create enhanced response prompt
        prompt = (
            f"Using the following medical knowledge:\n{knowledge_base}\n"
            f"Answer the question in a professional and medically accurate manner: {query}"
        )
        return self.llm.complete(prompt)

# Instantiate chatbot with OpenAI GPT-4
chatbot = MedicalChatbot(
    name="Medical_Chatbot",
    llm_config={
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
)
```

---

## **Deployment & Testing**
### **Step 7: Run the Chatbot Locally**

1. **Run AutoGen Chatbot with UI**
   ```bash
   autogenstudio ui --port 8082
   ```

2. **Interact with the chatbot in Python**
   ```python
   response = chatbot.generate_reply([{"role": "user", "content": "What are the symptoms of diabetes?"}])
   print(response)
   ```

---

## **Continuous Learning & Data Updates**
To **update your chatbot**, periodically add **new data** and **retrain the FAISS index**:

```python
# Add new data to FAISS
new_medical_qa = [
    {"question": "What is hypertension?", "answer": "Hypertension is high blood pressure that can lead to heart problems."},
    {"question": "What are the treatments for asthma?", "answer": "Asthma treatments include inhalers, bronchodilators, and avoiding triggers."}
]

# Convert new data to embeddings
new_embeddings = embedding_model.encode(
    [qa["question"] + " " + qa["answer"] for qa in new_medical_qa],
    convert_to_numpy=True
)

# Add to FAISS index
index.add(new_embeddings)
faiss.write_index(index, "data/medical_faiss_index")
```


