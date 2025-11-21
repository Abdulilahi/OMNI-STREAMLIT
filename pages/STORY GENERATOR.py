# app_garden_theme.py
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage,SystemMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4

# -----------------------------
# Setup
# -----------------------------
api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
if not api_key or not hf_token:
    st.error("API keys not found!")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

pdfmetrics.registerFont(TTFont('NotoSans', 'kahaaniSuno/NotoSans-Regular.ttf'))
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 50

# -----------------------------
# Helper Functions
# -----------------------------
def generate_story(age_group, story_tone, moral_theme, language, characters):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a creative story writer generating unique stories."),
        HumanMessage(content=f"Generate a story for {age_group} with characters {characters}, a {story_tone} story, with {moral_theme} moral theme, in {language} language, include title, simple and engaging.")
    ])
    messages = prompt.format_messages()
    response = llm.invoke(messages)
    return response.content

def generate_image_prompt(story_text):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Summarize stories into vivid, colorful image prompts."),
        HumanMessage(content=f"Create a concise, fun, colorful illustration prompt for this story:\n\n{story_text}")
    ])
    messages = prompt.format_messages()
    response = llm.invoke(messages)
    return response.content

def split_story(text, parts=4):
    paragraphs = text.split('\n\n') if '\n\n' in text else [text]
    chunk_size = max(1, len(paragraphs)//parts)
    return ["\n\n".join(paragraphs[i*chunk_size:(i+1)*chunk_size]) for i in range(parts)]

def create_alternating_pdf(story_parts, images, output="storybook.pdf"):
    doc = SimpleDocTemplate(output, pagesize=A4)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle('Custom', parent=styles['Normal'], fontName='NotoSans', fontSize=14, leading=18)
    story = []

    for i in range(len(story_parts)):
        img = Image(images[i])
        orig_width, orig_height = img.wrap(0,0)
        max_width = PAGE_WIDTH - 2*MARGIN
        max_height = PAGE_HEIGHT - 2*MARGIN
        scale = min(max_width/orig_width, max_height/orig_height)
        img.drawWidth = orig_width * scale
        img.drawHeight = orig_height * scale
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(PageBreak())

        para = Paragraph(story_parts[i], custom_style)
        story.append(para)
        story.append(PageBreak())

    doc.build(story)
    return output

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🎨 Kids Magical Story Generator", page_icon="🌈", layout="wide")

# Apply clean CSS theme
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000; /* pure black background */
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent;
}

h1, h2, h3, h4, h5, h6, p, div, span {
    color: white !important; /* make all text white */
}

.stButton>button {
    background-color: #ff7e5f;
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
}
.stButton>button:hover {
    background-color: #feb47b;
    color: black;
}

div.stMarkdown {
    background-color: rgba(0, 0, 0, 0.0); /* transparent */
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Magical Story Generator for Kids 🌈")
st.subheader("Create your own illustrated storybook in seconds ✨")

with st.expander("📝 Enter Story Details"):
    age_group = st.selectbox("👶 Age Group", ["infant", "children"], index=1)
    story_tone = st.selectbox("🎭 Story Tone", ["soft", "hard"], index=0)
    moral_theme = st.text_input("💖 Moral Theme", value="love")
    language = st.text_input("🌍 Language", value="english")
    characters = st.text_input("🧚 Character Names", value="Your own characters")
    generate_btn = st.button("✨ Generate Story & PDF!")

if generate_btn:
    with st.spinner("Generating story..."):
        story = generate_story(age_group, story_tone, moral_theme, language, characters)
    st.success("Story generated! 🎉")
    
    st.subheader("📖 Story Preview")
    st.markdown(f"<div>{story}</div>", unsafe_allow_html=True)

    story_parts = split_story(story, 4)

    with st.spinner("Creating illustrations..."):
        client = InferenceClient(provider="hf-inference", api_key=hf_token)
        images = []
        for i, part in enumerate(story_parts):
            prompt = generate_image_prompt(part)
            img_obj = client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")
            img_path = f"story_image_{i+1}.png"
            img_obj.save(img_path)
            images.append(img_path)
    st.success("Illustrations ready! 🖼")

    st.subheader("🖼 Story Illustrations")
    cols = st.columns(2)
    for i, img in enumerate(images):
        cols[i%2].image(img, use_container_width=True)

    with st.spinner("Creating PDF..."):
        pdf_file = create_alternating_pdf(story_parts, images)
    st.success("PDF created successfully! 📄")

    with open(pdf_file, "rb") as f:
        st.download_button("📥 Download Your Magical Storybook", f, file_name="MagicStoryBook.pdf")
