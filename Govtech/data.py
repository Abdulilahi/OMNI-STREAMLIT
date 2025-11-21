# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import Pinecone as LC_Pinecone
# from pinecone import Pinecone as PC

# load_dotenv()

# # --- Initialize ---
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "rag1"

# pc = PC(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# # --- Embedding model (must match your main app) ---
# embedder = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en",
#     model_kwargs={"device": "cpu"}
# )



# # --- Store in Pinecone ---
# vectordb = LC_Pinecone(index=index, embedding=embedder, text_key="text")
# vectordb.add_texts(texts)
# print("✅ Successfully seeded initial data into Pinecone!")


# loader=WebBaseLoader("https://nscpolteksby.ac.id/ebook/files/Ebook/Business%20Administration/ARMSTRONGS%20HANDBOOK%20OF%20HUMAN%20RESOURCE%20MANAGEMENT%20PRACTICE/26%20-%20Job-Role-%20Competency%20and%20Skills%20Analysis.pdf")

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader,JSONLoader
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

ploader=PyPDFLoader(file_path="26 - Job-Role- Competency and Skills Analysis.pdf")
qloader=PyPDFLoader(file_path="job-description-compendium-by-workable.pdf")
rloader=PyPDFLoader(file_path="resume-and-cover-letter-examples.pdf")
sloader=JSONLoader(file_path="job_roles.json",jq_schema='.',text_content=False)


loads=[]
loads=ploader.load()
loads.extend(qloader.load())
loads.extend(rloader.load())
loads.extend(sloader.load())
len(loads)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
chunks = text_splitter.split_documents(loads)
chunks

from langchain.vectorstores import FAISS
len(chunks)
texts = [doc.page_content for doc in chunks]
text=[]
for i in texts:
    texts=text_splitter.split_text(i)
    text.extend(texts)

embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectordb = FAISS.from_texts(text, embedder)
