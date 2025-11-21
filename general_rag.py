import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load secrets (Cloud) or .env (Local) ---
if "LANGCHAIN_API_KEY" in st.secrets:
    langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    load_dotenv()
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

# Optional LangSmith monitoring
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = langchain_api_key

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Output parser
parser = StrOutputParser()

# LCEL Chain
chain = prompt | llm | parser

# Function callable from Streamlit
def answer_query(question: str) -> str:
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
