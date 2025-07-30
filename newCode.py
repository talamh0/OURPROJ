import os
import uuid
import arabic_reshaper
from bidi.algorithm import get_display

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 0) Make sure you’ve installed:
#    pip install arabic_reshaper python-bidi

# 1) Configuration
file_path      = "/Users/mac/Desktop/myProj/Guide to Essential Cybersecurity Controls Implementation.txt"
persist_path   = "./chroma_gemini_store"
google_api_key = "hhhhh"

# reshaper for Arabic
reshaper = arabic_reshaper.ArabicReshaper()

# 2) Load & chunk the document
loader   = TextLoader(file_path, encoding="utf-8")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs     = splitter.split_documents(raw_docs)

# 3) (Optional) Print how many chunks you got
print(f"Total chunks: {len(docs)}\n")

# 4) Create embeddings via Gemini
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

# 5) Build & persist Chroma vectorstore
os.makedirs(persist_path, exist_ok=True)
ids = [str(uuid.uuid4()) for _ in docs]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory=persist_path,
    ids=ids,
)
vectorstore.persist()
print(f"✅ Chroma store persisted at {persist_path}")

# 6) Set up Gemini LLM + RetrievalQA
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=google_api_key
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 7) Run your query
query = "كيف يكون المراجعة والتدقيق الدوري للأمن السيبراني؟"
result = qa({"query": query})

# 8) Reshape & display the LLM’s answer
raw_answer    = result["result"]
reshaped_ans  = reshaper.reshape(raw_answer)
display_answer = get_display(reshaped_ans)

print("\nAnswer:\n")
print(display_answer)

# 9) Reshape & display each retrieved chunk
print("\nRetrieved chunks:\n")
for i, doc in enumerate(result["source_documents"], start=1):
    raw_chunk     = doc.page_content
    reshaped_chunk = reshaper.reshape(raw_chunk)
    display_chunk  = get_display(reshaped_chunk)

    print(f"[{i}] {display_chunk}\n")

# 10) (Optional) also just run simple .run() with reshaping
raw_simple = qa.run(query)
print("Q:", query)
print("A:", get_display(reshaper.reshape(raw_simple)))
