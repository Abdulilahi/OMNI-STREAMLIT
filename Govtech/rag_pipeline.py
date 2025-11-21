
import os
from dotenv import load_dotenv
from langchain_anthropic import AnthropicLLM
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()
import requests
from bs4 import BeautifulSoup
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
nvidia_api_key=os.getenv("NVIDIA_API_KEY")


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = langchain_api_key

from langchain_tavily import TavilySearch

# Initialize search tool
search = TavilySearch(max_results=10, tavily_api_key=os.getenv("TAVILY_API_KEY"))

def collect_urls(query: str):
    """Collect URLs from Tavily search results."""
    results = search.invoke(query)   # Tavily gives a dict
    print(results)
    urls = [r["url"] for r in results["results"] if "url" in r]
    print(urls)
    return urls


# --- Scraper ---
def scrape_url(url):
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # remove unwanted tags
        for s in soup(["script", "noscript", "style"]):
            s.decompose()
        text_tags = soup.find_all([
            "h1","h2","h3","h4","h5","h6",
            "p","li","blockquote","span","a","div"
        ])
        # extract only <p> tags
        content = [t.get_text(strip=True) for t in text_tags if t.get_text(strip=True)]

        if not content:
            return "⚠️ No useful tags found."

        # return first 50 chunks for readability
        return "\n\n".join(content[:300])

    except Exception as e:
        return f"❌ Failed to scrape {url}: {e}"




def build_dynamic_vectorstore(query: str):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = [doc.page_content for doc in docs]
    txt=[]
    for i in texts[:10]:
        text=text_splitter.split_text(i);
        txt.extend(text)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_texts(txt, embedder)
    return vectordb

vectordb=build_dynamic_vectorstore("What does source explains about?")
retriever=vectordb.as_retriever(search_kwargs={"k":3})
prompt=ChatPromptTemplate.from_messages([
    ("human","Answer the following Question based on the following {context} Question:{input}")
    
])
doc_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

result = qa_chain.invoke({"input": "What does source explains about?"})
print(result["answer"])
