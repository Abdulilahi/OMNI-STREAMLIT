import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Load env vars
load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Init Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Optional: enable tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = langchain_api_key

# 🔹 Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# 🔹 LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# ✅ Function that your Streamlit app can call
def answer_query(question: str) -> str:
    try:
        response = chain.run({"question": question})
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
