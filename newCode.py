import os
import uuid
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# 1) Configuration
file_path = "/Users/mac/Desktop/myProj/71-98.txt"
persist_path = "./chroma_gemini_store"

# 2) Load & chunk the document

loader = TextLoader(file_path, encoding="utf-8")
raw_docs = loader.load()  # returns List[Document]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)

# 3) Create embeddings via Gemini

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="google_api_key"

)  # :contentReference[oaicite:0]{index=0}

# 4) Build & persist Chroma vectorstore

# ensure directory exists
os.makedirs(persist_path, exist_ok=True)

# generate stable IDs for each chunk
ids = [str(uuid.uuid4()) for _ in docs]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory=persist_path,
    ids=ids,
)
vectorstore.persist()
print(f"✅ Chroma store persisted at {persist_path}")

# 5) Set up Gemini LLM + RetrievalQA

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key="google_api_key"
)  # :contentReference[oaicite:1]{index=1}

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# 6) Example query
question = "ما ارشادات تطبيق الضوابط ١-٣-٧-٢ لملكية البيانات و المعلومات؟"
print("Q:", question)
print("A:", qa.run(question))
