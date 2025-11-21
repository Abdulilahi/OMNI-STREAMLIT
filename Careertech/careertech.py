import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
import time
from dotenv import load_dotenv
from typing import List

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from pinecone import Pinecone as PC, ServerlessSpec
from langchain_tavily import TavilySearch
load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedder = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}
# )

# embedder = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en",
#     model_kwargs={"device": "cpu", "low_cpu_mem_usage": False}
# )
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}
)

vectordb = FAISS.load_local("Careertech/faiss_index1", embedder, allow_dangerous_deserialization=True)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key,
)

search = TavilySearch(max_results=10, tavily_api_key=st.secrets["TAVILY_API_KEY"])

def collect_urls(query: str) -> List[str]:
    results = search.invoke(query)
    urls = [r["url"] for r in results.get("results", []) if "url" in r]
    return urls

# --- Scraper ---
def scrape_url(url: str) -> str:
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # remove junk
        for s in soup(["script", "noscript", "style"]):
            s.decompose()

        text_tags = soup.find_all([
            "h1","h2","h3","h4","h5","h6",
            "p","li","blockquote"
        ])
        content = [t.get_text(" ", strip=True) for t in text_tags if t.get_text(strip=True)]
        return "\n\n".join(content[:100]) if content else ""
    except Exception as e:
        return f"❌ Failed to scrape {url}: {e}"

def update_vectorstore(query: str):
    urls = collect_urls(query)
    scraped = [scrape_url(u) for u in urls[:3] if u]

    if not scraped:
        return

    raw_text = "\n\n".join(scraped)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    cleaned_chunks = []
    for chunk in chunks:
        try:
            resp = llm.invoke(f"Clean and structure this text:\n\n{chunk}")
            cleaned_chunks.append(resp.content)
        except Exception as e:
            print(f"⚠️ Skipping chunk due to error: {e}")
            continue

    if cleaned_chunks:
        vectordb.add_texts(cleaned_chunks)
        time.sleep(1)


# --- QA ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful career advise assistant "),
    ("human", "Answer strictly based on this context:\n\n{context}\n\nQuestion: {input} if you are unsure or  not found any  related content in context strictly say 'I don't know'")
])

def answer_query(query: str) -> str:
    # 1) Try to answer directly from DB
    chain = create_retrieval_chain(
        vectordb.as_retriever(search_kwargs={"k": 3}),
        create_stuff_documents_chain(llm, prompt)
    )
    result = chain.invoke({"input": query})
    answer = result.get("answer") or result.get("output_text") or ""

    # Normalize
    answer_lower = answer.lower().strip()
    no_info_flags = ["no information", 'didn\'t',"did not","don\'t", "not available", "i don’t know", "idontknow","unable"]

    if answer_lower == "" or any(flag in answer_lower for flag in no_info_flags):
        print("⚠️ Answer looks like 'no info' → Scraping...")
        update_vectorstore(query)

        # Retry after scraping
        result = chain.invoke({"input": query})
        answer = result.get("answer") or result.get("output_text") or ""
        if answer.strip():
            return answer
        return "⚠️ Sorry, nothing relevant found even after scraping."

    return answer

    return answer

# --- Main loop ---
if __name__ == "__main__":
    while True:
        query = input("Ask a career question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        print("\n📝 Answer:", answer_query(query), "\n")



# import os
# import requests
# from bs4 import BeautifulSoup
# import time
# from dotenv import load_dotenv
# from typing import List

# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain.chains import create_retrieval_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_tavily import TavilySearch

# # --- Load environment variables ---
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# tavily_api_key = os.getenv("TAVILY_API_KEY")

# # --- Initialize OpenAI embeddings ---
# # embedder = OpenAIEmbeddings(
# #     model="text-embedding-3-small",  # or "text-embedding-3-large" for better accuracy
# #     api_key=openai_api_key
# # )


# # --- Load FAISS vectorstore ---
# vectordb = FAISS.load_local("Careertech/faiss_index", embedder, allow_dangerous_deserialization=True)

# # --- Initialize Groq LLM ---
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     api_key=groq_api_key,
# )

# # --- Tavily Search ---
# search = TavilySearch(max_results=10, tavily_api_key=tavily_api_key)

# # --- Collect URLs ---
# def collect_urls(query: str) -> List[str]:
#     results = search.invoke(query)
#     urls = [r["url"] for r in results.get("results", []) if "url" in r]
#     return urls

# # --- Scraper ---
# def scrape_url(url: str) -> str:
#     DEFAULT_HEADERS = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                       "AppleWebKit/537.36 (KHTML, like Gecko) "
#                       "Chrome/120.0.0.0 Safari/537.36"
#     }
#     try:
#         r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
#         r.raise_for_status()
#         soup = BeautifulSoup(r.text, "html.parser")

#         for s in soup(["script", "noscript", "style"]):
#             s.decompose()

#         text_tags = soup.find_all([
#             "h1","h2","h3","h4","h5","h6",
#             "p","li","blockquote"
#         ])
#         content = [t.get_text(" ", strip=True) for t in text_tags if t.get_text(strip=True)]
#         return "\n\n".join(content[:100]) if content else ""
#     except Exception as e:
#         return f"❌ Failed to scrape {url}: {e}"

# # --- Update Vector Store ---
# def update_vectorstore(query: str):
#     urls = collect_urls(query)
#     scraped = [scrape_url(u) for u in urls[:3] if u]

#     if not scraped:
#         return

#     raw_text = "\n\n".join(scraped)

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
#     chunks = text_splitter.split_text(raw_text)

#     cleaned_chunks = []
#     for chunk in chunks:
#         try:
#             resp = llm.invoke(f"Clean and structure this text:\n\n{chunk}")
#             cleaned_chunks.append(resp.content)
#         except Exception as e:
#             print(f"⚠️ Skipping chunk due to error: {e}")
#             continue

#     if cleaned_chunks:
#         vectordb.add_texts(cleaned_chunks)
#         # Optional: persist updated DB
#         vectordb.save_local("Careertech/faiss_index")
#         time.sleep(1)

# # --- Prompt for QA ---
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful career advisor."),
#     ("human", "Answer strictly based on this context:\n\n{context}\n\nQuestion: {input}\n\nIf nothing relevant is found, say 'I don't know'.")
# ])

# # --- Answer Query ---
# def answer_query(query: str) -> str:
#     chain = create_retrieval_chain(
#         vectordb.as_retriever(search_kwargs={"k": 3}),
#         create_stuff_documents_chain(llm, prompt)
#     )
#     result = chain.invoke({"input": query})
#     answer = result.get("answer") or result.get("output_text") or ""

#     answer_lower = answer.lower().strip()
#     no_info_flags = ["no information", "not available", "i don’t know", "idontknow", "unable"]

#     if answer_lower == "" or any(flag in answer_lower for flag in no_info_flags):
#         print("⚠️ Answer looks like 'no info' → Scraping...")
#         update_vectorstore(query)
#         result = chain.invoke({"input": query})
#         answer = result.get("answer") or result.get("output_text") or ""
#         if answer.strip():
#             return answer
#         return "⚠️ Sorry, nothing relevant found even after scraping."

#     return answer

# # --- Main Loop ---
# if __name__ == "__main__":
#     while True:
#         query = input("Ask a career question (or 'exit' to quit): ")
#         if query.lower() == "exit":
#             break
        
#         print("\n📝 Answer:", answer_query(query), "\n")
