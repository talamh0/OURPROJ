import os
import uuid
import shutil
import nltk
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import google.generativeai as genai

# Set up Google Generative AI API
genai.configure(api_key="AIzaSyD55FBUYSZ2ZkzhK7WAokfsZdzD8Ddw_L8")

# Define file paths
file_paths = [
    "/Users/mac/Desktop/myProj/Guide to Essential Cybersecurity Controls Implementation.txt",
    
]

# Load files
docs = []
for path in file_paths:
    if not os.path.exists(path.strip()):
        print(f"File not found: {path}")
        continue
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path.strip())
    elif path.endswith(".txt"):
        loader = TextLoader(path.strip(), encoding="utf-8")
    else:
        print(f"Unsupported file format: {path}")
        continue

    try:
        loaded = loader.load()
        print(f"Loaded: {path} — Document count: {len(loaded)}")
        docs.extend(loaded)
    except Exception as e:
        print(f"Failed to load file {path}: {e}")

print(f"\nTotal documents loaded: {len(docs)}")

# Clean text
nltk.download("punkt")
for doc in docs:
    doc.page_content = doc.page_content.replace("\n", " ").replace("  ", " ").strip()

# Split into sentences, then into chunks
all_text = "\n".join([doc.page_content for doc in docs])
sentences = sent_tokenize(all_text)
print(f"Total sentences: {len(sentences)}")

chunk_size = 5
overlap = 2
chunks = []
i = 0
while i < len(sentences):
    chunk = sentences[i:i + chunk_size]
    chunks.append(" ".join(chunk))
    i += chunk_size - overlap

documents = [Document(page_content=chunk, metadata={"id": str(uuid.uuid4())}) for chunk in chunks]
print(f"Total chunks generated: {len(documents)}")

# Check for matching content (optional debug)
matched_chunks = [doc for doc in documents if "الحوسبة السحابية" in doc.page_content]
print(f"Chunks containing 'الحوسبة السحابية': {len(matched_chunks)}")
for i, doc in enumerate(matched_chunks[:3]):
    print(f"\nChunk {i+1}:\n{doc.page_content}\n{'-'*40}")

# Delete existing Chroma store (if exists)
chroma_base_path = "chroma_store"
if os.path.exists(chroma_base_path):
    shutil.rmtree(chroma_base_path)
    print("Old Chroma store deleted.")

persist_path = chroma_base_path

# Generate embeddings and build new Chroma store
embeddings_store = []
for i, doc in enumerate(documents):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=doc.page_content,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings_store.append({
            "chunk": doc.page_content,
            "embedding": result["embedding"]
        })
        print(f"Chunk {i+1} embedded.")
    except Exception as e:
        print(f"Embedding failed for chunk {i+1}: {e}")

# Custom embedding wrapper
class ManualEmbeddingModel(Embeddings):
    def __init__(self, vectors_dict):
        self.vectors_dict = vectors_dict

    def embed_documents(self, texts):
        return [self.vectors_dict[text] for text in texts]

    def embed_query(self, text):
        return self.vectors_dict.get(text, [0.0] * 768)

# Build document store and vector map
vectors_dict = {}
indexed_documents = []
for item in embeddings_store:
    text = item["chunk"]
    doc_id = str(uuid.uuid4())
    indexed_documents.append(Document(page_content=text, metadata={"id": doc_id}))
    vectors_dict[text] = item["embedding"]

embedding_model = ManualEmbeddingModel(vectors_dict)
vectorstore = Chroma.from_documents(
    documents=indexed_documents,
    embedding=embedding_model,
    persist_directory=persist_path
)
vectorstore.persist()
print("Chroma store successfully created and saved.")

# Quick verification
print("\nQuick Check:\n")
if documents:
    print("First 3 chunks:\n")
    for i, doc in enumerate(documents[:3]):
        print(f"Chunk {i+1}:\n{doc.page_content[:300]}\n{'-'*50}")
else:
    print("No chunks found.")

print("\nSetup complete.")
