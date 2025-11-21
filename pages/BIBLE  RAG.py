import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.cassandra import Cassandra
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from langchain_huggingface import HuggingFaceEmbeddings
# Load environment variables
load_dotenv()

# Keys
ASTRA_DB_APPLICATION_TOKEN_BIBLE = os.getenv("ASTRA_DB_APPLICATION_TOKEN_BIBLE")
ASTRA_DB_ID_BIBLE = os.getenv("ASTRA_DB_ID_BIBLE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN_BIBLE, database_id=ASTRA_DB_ID_BIBLE)

# embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")


astra_vector_store = Cassandra(
    embedding=embedder,
    table_name="qa_mini",
    session=None,
    keyspace=None,
)

# Insert docs only if empty (to avoid duplicates each run)
# if astra_vector_store._collection.count_documents({}) == 0:
#     astra_vector_store.add_documents(splits)

retriever = astra_vector_store.as_retriever(search_kwargs={"k": 4})

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a Bible assistant. Use only the provided context to answer the question. answer the question correctly understand it analyse it wth the context,
If the answer is not in the context, say: "I could not find this in the document."

Context:
{context}

Question: {input}
Answer:
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# --- Streamlit UI ---
st.set_page_config(page_title="Bible Assistant", page_icon="📖", layout="wide")
st.title("📖 Bible Assistant")
st.caption("Ask me anything from the Bible. I will answer based only on the text i trained on.")
import base64
def set_bg(png_file):
    with open("5.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        """,
        unsafe_allow_html=True
    )

set_bg("7.png")
if "messages" not in st.session_state:
    st.session_state["messages"] = []





# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask a question about the Bible..."):
    # Save & show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bot response
    with st.spinner("🔎 Searching for the answer..."):
        response = retrieval_chain.invoke({"input": user_input})
        bot_reply = response["answer"]

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
