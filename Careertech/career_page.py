import streamlit as st


from Careertech.careertech import answer_query as career_answer
import base64

# st.set_page_config(page_title="CareerTech Chat", layout="wide")
st.title("💼 CareerTech Assistant")

@st.cache_data
def get_bg_encoded(png_file):
    with open(png_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_bg(png_file):
    encoded = get_bg_encoded(png_file)
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

set_bg("3.png")

if "career_history" not in st.session_state:
    st.session_state.career_history = []

query = st.chat_input("Ask about careers...")

if query:
    st.session_state.career_history.append({"role": "user", "content": query})
    answer = career_answer(query)
    st.session_state.career_history.append({"role": "assistant", "content": answer})

for msg in st.session_state.career_history:
    st.chat_message(msg["role"]).write(msg["content"])
