# import streamlit as st
# from general_rag import answer_query as general_answer

# st.set_page_config(page_title="🌐 OMNI RAG", layout="wide")

# # ---- Sidebar Navigation ----
# st.sidebar.title("📂 Navigation")

# # Personal
# st.sidebar.markdown("### 👤 Personal")
# if st.sidebar.button("💼 CareerTech"):
#     st.switch_page("pages/1_CareerTech.py")

# if st.sidebar.button("🏛️ GovTech"):
#     st.switch_page("pages/2_GovTech.py")

# if st.sidebar.button("🎨 Kids StoryBook Generator"):
#     st.switch_page("pages/5_story_generator.py")


# # Mythological
# st.sidebar.markdown("### 📖 Mythological")
# if st.sidebar.button("📿 Quran Assistant"):
#     st.switch_page("pages/qura.py")

# if st.sidebar.button("🕉️ Bhavadgita Assistant"):
#     st.switch_page("pages/gita.py")

# if st.sidebar.button("✝️ Bible Assistant"):
#     st.switch_page("pages/bib.py")


# # Extra Tools
# st.sidebar.markdown("### 🛠️ Extra Tools")
# if st.sidebar.button("📄 PDF Explainer"):
#     st.switch_page("pages/3_PDF_Explainer.py")

# if st.sidebar.button("📄 PDF Summarizer"):
#     st.switch_page("pages/4_pdf_summariser.py")


# # ---- Main Page (General RAG only) ----
# st.title("🌐 OMNI RAG")
# st.markdown("### 🔍 Ask me anything (general_rag.py)")

# query = st.text_input("Type your question here...")
# if query:
#     try:
#         answer = general_answer(query)
#         st.success(f"📝 {answer}")
#     except Exception as e:
#         st.error(f"⚠️ Error: {e}")





# import streamlit as st
# from general_rag import answer_query as general_answer

# st.set_page_config(page_title="🌐 OMNI RAG", layout="wide")

# # ---- Centered Title ----
# st.markdown("<h1 style='text-align: center;'>🌐 OMNI RAG</h1>", unsafe_allow_html=True)

# # ---- RAG Assistants Section ----
# st.markdown("### 🚀 Quick Access to RAG Assistants")

# col1, col2, col3 = st.columns(3)
# with col1:
#     if st.button("💼 CareerTech"):
#         st.switch_page("pages/1_CareerTech.py")
#     if st.button("🏛️ GovTech"):
#         st.switch_page("pages/2_GovTech.py")
#     if st.button("🎨 Kids StoryBook Generator"):
#         st.switch_page("pages/5_story_generator.py")

# with col2:
#     if st.button("📿 Quran Assistant"):
#         st.switch_page("pages/qura.py")
#     if st.button("🕉️ Bhavadgita Assistant"):
#         st.switch_page("pages/gita.py")
#     if st.button("✝️ Bible Assistant"):
#         st.switch_page("pages/bib.py")

# with col3:
#     if st.button("📄 PDF Explainer"):
#         st.switch_page("pages/3_PDF_Explainer.py")
#     if st.button("📄 PDF Summarizer"):
#         st.switch_page("pages/4_pdf_summariser.py")

# # ---- Spacer ----
# st.markdown("---")

# # ---- General RAG in Center ----
# st.markdown("### 🔍 General RAG", unsafe_allow_html=True)
# query = st.text_input("Ask me anything:", key="general_query", placeholder="Type your question here...")

# if query:
#     try:
#         answer = general_answer(query)
#         st.success(f"📝 {answer}")
#     except Exception as e:
#         st.error(f"⚠️ Error: {e}")


# import streamlit as st
# from general_rag import answer_query as general_answer
# import base64

# st.set_page_config(page_title="🌐 OMNI RAG", layout="wide")
# # 
# # ---- Background Image ----
# def set_bg(png_file):
#     with open("bg.png", "rb") as f:
#         encoded = base64.b64encode(f.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
#             background-size: cover;
#         }}
#         .stButton>button {{
#             border-radius: 25px;
#             padding: 0.6em 1.2em;
#             font-size: 16px;
#             font-weight: 600;
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
#             transition: 0.3s;
#         }}
#         .stButton>button:hover {{
#             background-color: #45a049;
#             transform: scale(1.05);
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# set_bg("bg.png")

# # ---- Centered Title ----
# # st.markdown("<h1 style='text-align: center; color: white;'>🌐 OMNI RAG</h1>", unsafe_allow_html=True)


# # ---- RAG Assistants Section ----
# st.markdown("### 🚀 Quick Access to RAG Assistants")

# col1, col2, col3 = st.columns(3)
# with col1:
#     if st.button("💼 CareerTech"):
#         st.switch_page("pages/1_CareerTech.py")
#     if st.button("🏛️ GovTech"):
#         st.switch_page("pages/2_GovTech.py")
#     if st.button("🎨 Kids StoryBook Generator"):
#         st.switch_page("pages/5_story_generator.py")

# with col2:
#     if st.button("📿 Quran Assistant"):
#         st.switch_page("pages/qura.py")
#     if st.button("🕉️ Bhavadgita Assistant"):
#         st.switch_page("pages/gita.py")
#     if st.button("✝️ Bible Assistant"):
#         st.switch_page("pages/bib.py")

# with col3:
#     if st.button("📄 PDF Explainer"):
#         st.switch_page("pages/3_PDF_Explainer.py")
#     if st.button("📄 PDF Summarizer"):
#         st.switch_page("pages/4_pdf_summariser.py")


# # ---- General RAG in Center ----
# st.markdown("<h3 style='text-align: center; color: white;'>🔍 General RAG</h3>", unsafe_allow_html=True)
# query = st.text_input("Ask me anything:", key="general_query", placeholder="Type your question here...")

# if query:
#     try:
#         answer = general_answer(query)
#         st.success(f"📝 {answer}")
#     except Exception as e:
#         st.error(f"⚠️ Error: {e}")


import streamlit as st
from general_rag import answer_query as general_answer
import base64

st.set_page_config(page_title="🌐 OMNI RAG", layout="wide")

# ---- Background Image ----
def set_bg(png_file):
    with open("2.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Move Assistants Section Down */
        .assistants {{
            margin-top: 200px; /* ~5cm */
        }}

        /* Neon Button Styling */
        /*.stButton>button {{
            border-radius: 25px;
            padding: 0.7em 1.4em;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(90deg, #00f260, #0575e6);
            color: white;
            border: none;
            box-shadow: 0px 0px 12px rgba(0, 255, 180, 0.8);
            transition: 0.3s;
        }}*/
        .stButton>button:hover {{
            /* background: linear-gradient(30deg, #ebbba7, #cfc7f8);*/
           /*background: linear-gradient(90deg, #A8C0CB, #394E5A);*/ /* Slate & Mist gradient */
    transform: scale(1.07);
    /* Adjust box-shadow color to match the new subtle colors, e.g., a subtle grey or pale blue */
    box-shadow: 0px 0px 15px rgba(200, 200, 200, 0.7); 
        }}

        /* Rounded General RAG Input */
        div[data-baseweb="input"] > div {{
            border-radius: 25px;
            box-shadow: 0px 0px 10px rgba(0, 255, 200, 0.7);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.png")

# ---- Centered Title ----
# st.markdown("<h1 style='text-align: center; color: white;'>🌐 OMNI RAG</h1>", unsafe_allow_html=True)

# ---- General RAG in Center ----
st.markdown("<h3 style='text-align: center; color: white;font-size:50px'>🔍 OMNI MULTI DOMAIN RAG CHATBOT SYSTEM</h3>", unsafe_allow_html=True)
query = st.text_input("Ask me anything:", key="general_query", placeholder="Type your question here...")

if query:
    try:
        answer = general_answer(query)
        st.success(f"📝 {answer}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")


# ---- RAG Assistants Section ----
st.markdown("<h3 class='assistants' style='text-align: center;color: white;padding-top:90px;padding-bottom:30px;'>🚀 QUICK ACCESS TO OMNIRAG ASSISTANTS</h3>", unsafe_allow_html=True)
# st.markdown("<h3 style=' color: white;'>🔍 General RAG</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💼 CAREERTECH RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/CAREERTECH_RAG.py")
    if st.button("🏛️ GOV POLICY RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/GOV POLICY RAG.py")
    if st.button("🎨 KIDS STORY GENERATOR",use_container_width=True):
        st.switch_page("pages/STORY GENERATOR.py")

with col2:
    if st.button("🌙 QURAN_RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/QURAN_RAG.py")
    if st.button("🕉️ BHAVAD GITA RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/BHAVAD GITA.py")
    if st.button("✝️ Bible RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/BIBLE RAG.py")

with col3:
    if st.button("📝 PDF EXPLAINER RAG",use_container_width=True):
        st.switch_page("pages/PDF_EXPLAINER.py")
    if st.button("📋 PDF SUMMARIZER RAG",use_container_width=True):
        st.switch_page("pages/PDF_SUMMARISER.py")
    if st.button("📐 MATHS RAG ASSISTANT",use_container_width=True):
        st.switch_page("pages/MATHS RAG.py")

