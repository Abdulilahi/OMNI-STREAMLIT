import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient



st.title("📄 PDF Summarizer + 🔊 Text-to-Speech")
import base64
def set_bg(png_file):
    with open("4.png", "rb") as f:
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

set_bg("4.png")
# API Key Inputs
openai_key = st.text_input("Enter your OpenAI API Key", type="password")
hf_token = st.text_input("Enter your HuggingFace Token", type="password")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if openai_key and uploaded_file:
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
        openai_api_key=openai_key
    )

    # Generate structured text
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing..."):
            response = llm.invoke(
                f"Make all the text in a structured manner. Text follows:\n{a}\n\nGive it back in paragraphs."
            )
        summary_text = response.content
        st.subheader("📑 Structured Text")
        st.write(summary_text)

        # --- Text-to-Speech Section ---
        if hf_token:
            # client = InferenceClient(provider="fal-ai", api_key=hf_token)
            client = InferenceClient(provider="replicate", api_key=hf_token)
            if st.button("Convert Summary to Speech"):
                with st.spinner("Generating speech..."):
                    # audio = client.text_to_speech(
                    #     summary_text,
                    #     model="hexgrad/Kokoro-82M"
                    # )
                    audio = client.audio_to_audio(summary_text,
                        model="jaaari/kokoro-82m")   

# Save and play back
                with open("output.wav", "wb") as f:
                    f.write(audio[0].bytes)

                st.audio("output.wav", format="audio/wav")
                st.success("✅ Speech generated successfully!")

        else:
            st.warning("⚠️ Enter your HuggingFace Token to enable Text-to-Speech.")

elif not openai_key:
    st.warning("⚠️ Please enter your OpenAI API Key.")
elif not uploaded_file:
    st.warning("⚠️ Please upload a PDF file.")
