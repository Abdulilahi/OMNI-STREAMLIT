import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

st.title("📄 PDF Summarizer with OpenAI")
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

set_bg("5.png")
# API Key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if api_key and uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    ploader = PyPDFLoader("temp.pdf")
    pages = ploader.load()
    text = [p.page_content for p in pages]

    # Summariser
    def summariser(text_list):
        return "\n".join(text_list)

    a = summariser(text)

    # LLM setup
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key
    )

    # Generate structured text
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing..."):
            response = llm.invoke(
                f"Make all the text in a structured manner. Text follows:\n{a}\n\nGive it back in paragraphs."
            )
        st.subheader("📑 Structured Text")
        st.write(response.content)

elif not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key.")
elif not uploaded_file:
    st.warning("⚠️ Please upload a PDF file.")
