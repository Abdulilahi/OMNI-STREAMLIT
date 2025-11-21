import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Enable LangSmith (optional)
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

# Build chain using LCEL
chain = prompt | llm | parser

# Function to call from Streamlit
def answer_query(question: str) -> str:
    try:
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
