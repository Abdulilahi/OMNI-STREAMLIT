import os
import requests
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
from typing import List
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LC_Pinecone
from pinecone import Pinecone as PC, ServerlessSpec
from langchain_tavily import TavilySearch
from langchain_openai import OpenAIEmbeddings
# --- ENV ---
load_dotenv()
groq_api_key = st.secrets["GROQ_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "rag"

# --- LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
openai_api_key = st.secrets["OPENAI_API_KEY"]


embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}
)

 
from pinecone import Pinecone as PC, ServerlessSpec
from langchain_pinecone import Pinecone as LC_Pinecone
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "us-east-1"  # choose your region
INDEX_NAME = "rag1"

pc=PC(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)
vectordb = LC_Pinecone(index=index,
    embedding=embedder,  
    text_key="text") 
# --- Tavily search ---
search = TavilySearch(max_results=10, tavily_api_key=st.secrets["TAVILY_API_KEY"])

def collect_urls(query: str) -> List[str]:
    print(f"🔍 Tavily search for query: {query}")
    results = search.invoke(query)
    urls = [r["url"] for r in results.get("results", []) if "url" in r]
    print(f"✅ Got {len(urls)} URLs")
    return urls

# --- Scraper ---
def scrape_url(url: str) -> str:
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        print(f"🌐 Scraping: {url}")
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for s in soup(["script", "noscript", "style"]):
            s.decompose()

        text_tags = soup.find_all([
            "h1","h2","h3","h4","h5","h6",
            "p","li","blockquote"
        ])
        content = [t.get_text(" ", strip=True) for t in text_tags if t.get_text(strip=True)]
        text = "\n\n".join(content[:100]) if content else ""
        print(f"📄 Scraped {len(text)} characters")
        return text
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

def update_vectorstore(query: str):
    urls = collect_urls(query)
    scraped = [scrape_url(u) for u in urls[:3] if u]

    if not scraped:
        print("⚠️ No pages scraped")
        return

    raw_text = "\n\n".join(scraped)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)
    print(f"✂️ Split into {len(chunks)} chunks")
    cleaned_chunks = []
    for c in chunks:
        resp = llm.invoke(f"Clean this text:\n\n{c}")
        if hasattr(resp, "content"):
            cleaned_chunks.append(resp.content)
        else:
            cleaned_chunks.append(str(resp))
    # Store raw chunks (skip Groq cleaning for now)
    vectordb.add_texts(cleaned_chunks)
    print("📥 Stored chunks in Pinecone")

# --- QA ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for government policy questions."),
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
    no_info_flags =  ["no information", 'didn\'t',"did not","don\'t", "not available", "i don’t know", "idontknow","unable","unfortunately","do not"]

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

# --- Main loop ---
if __name__ == "__main__":
    while True:
        query = input("Ask a government question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print("\n📝 Answer:", answer_query(query), "\n")