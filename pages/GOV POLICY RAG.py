import streamlit as st
from Govtech.govttech import answer_query as gov_answer

st.set_page_config(page_title="GovTech Chat", layout="wide")

st.title("🏛️ GovTech Assistant")
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
if "gov_history" not in st.session_state:
    st.session_state.gov_history = []

query = st.chat_input("Ask about government policies...")

if query:
    st.session_state.gov_history.append({"role": "user", "content": query})
    answer = gov_answer(query)
    st.session_state.gov_history.append({"role": "assistant", "content": answer})

for msg in st.session_state.gov_history:
    st.chat_message(msg["role"]).write(msg["content"])
