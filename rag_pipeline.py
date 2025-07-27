# rag_pipeline.py

import os
from dotenv import load_dotenv
import requests
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredPDFLoader
from bidi.algorithm import get_display
import arabic_reshaper


# 🔐 Load your API key (not hardcoded in production!)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key # Replace with your actual key

# 🔮 Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 🧠 Memory and chat prompt
memory = ConversationBufferMemory(return_messages=True)
chat_prompt = PromptTemplate.from_template(
    """You are a helpful, professional AI named Bomba.
    You are responsible for helping cybersecurity companies understand national cyber policies. Do your best to answer the question.

Chat History:
{history}
User: {input}
Bomba:"""
)

chatbot = ConversationChain(llm=llm, memory=memory, prompt=chat_prompt, verbose=False)

# 🛠 Tools for agent
def current_time_tool(_):
    return f"📅 Current datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def fetch_example_dot_com(_):
    response = requests.get("https://example.com")
    return response.text[:1000]

def safe_calculator(query: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if any(c not in allowed for c in query):
            return "Only math expressions are allowed."
        return str(eval(query))
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    PythonREPLTool(),
    Tool(name="Calculator", func=safe_calculator, description="Basic math operations."),
    Tool(name="CurrentTime", func=current_time_tool, description="Returns current datetime."),
    Tool(name="FetchExampleDotCom", func=fetch_example_dot_com, description="Fetches example.com content."),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 📚 RAG Setup
loader = TextLoader( "/Users/mac/Desktop/my-project/trial1.txt")

docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="rag-kb")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 🧪 Sample Queries
if __name__ == "__main__":
    response = rag_chain.run("ما هي الارشادات الرئيسية لمستهلكي التجارة الالكترونية؟؟")
    reshaped_text = arabic_reshaper.reshape(response)
    bidi_text = get_display(reshaped_text)
    print("👉 RAG response:", bidi_text)

