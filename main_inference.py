import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings

# Set up Google API key
genai.configure(api_key="API_key")

# Define custom embedding class using Gemini embeddings
class ManualEmbeddingModel(Embeddings):
    def __init__(self):
        pass

    def embed_documents(self, texts):
        return [
            genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )["embedding"]

# Initialize the LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyD55FBUYSZ2ZkzhK7WAokfsZdzD8Ddw_L8"
)

# Memory and Prompt (Arabic-focused, formal)
memory = ConversationBufferMemory(return_messages=True)

chat_prompt = PromptTemplate.from_template("""
أنت مساعد ذكي ومتخصص في السياسات السيبرانية وحماية المستهلك.

**تعليمات الإجابة:**
* اعتمد كليًا على المعلومات المقدمة في الوثائق الرسمية فقط.
* أجب باللغة العربية الرسمية بأسلوب دقيق وواضح.
* تجنب التخمين أو تقديم معلومات غير موثقة.
* **إذا لم تتوفر لديك الإجابة بناءً على الوثائق المتاحة، فاذكر بوضوح أنك لا تعرف الإجابة وأن الأفضل هو الرجوع إلى المسؤولين المختصين.**

**سياق المحادثة السابق:**
{history}

**سؤال المستخدم:**
{input}

**الرد:**
""")
# Load persistent Chroma vector store
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=ManualEmbeddingModel()
)

# Convert Chroma to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create Retrieval QA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Start interactive Q&A session
print("\nYou can now ask questions (type 'exit' to end the session):\n")

while True:
    question = input("Your question: ").strip()
    
    if question.lower() in ["exit", "quit", "توقف"]:
        print("\nSession ended.")
        break

    try:
        response = rag_chain.invoke({"query": question})
        print(f"\nAnswer:\n{response['result']}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
