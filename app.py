
import os
import base64
from datetime import datetime

import streamlit as st
import pymupdf as fitz
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

st.set_page_config(
    page_title="RAG Eco-Tech Assistant",
    page_icon="🌿",
    layout="wide"
)

# ---------------- EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def extract_pdf_text(uploaded_file):
    with fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    ) as document:
        return "\n".join(page.get_text() for page in document)


@st.cache_data(show_spinner=False)
def create_embeddings(chunks):
    model = load_embedding_model()
    vectors = model.encode(
        list(chunks),
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.asarray(vectors, dtype=np.float32)


# ---------------- ECO-TECH BACKGROUND ----------------
# Embedded SVG: digital globe, wind turbines, solar panels,
# rolling green hills and leaves. No extra image file required.

svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000"
viewBox="0 0 1600 1000">
<defs>
 <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
  <stop stop-color="#b9e8f0"/>
  <stop offset="0.55" stop-color="#e7f6eb"/>
  <stop offset="1" stop-color="#c6e9d2"/>
 </linearGradient>
 <linearGradient id="hill" x1="0" y1="0" x2="0" y2="1">
  <stop stop-color="#74c69d"/>
  <stop offset="1" stop-color="#23866a"/>
 </linearGradient>
 <radialGradient id="globe">
  <stop stop-color="#d9fff0" stop-opacity=".95"/>
  <stop offset="1" stop-color="#12a69a" stop-opacity=".38"/>
 </radialGradient>
</defs>
<rect width="1600" height="1000" fill="url(#sky)"/>
<g fill="#fff" opacity=".5">
 <ellipse cx="270" cy="140" rx="190" ry="48"/>
 <ellipse cx="365" cy="125" rx="100" ry="60"/>
 <ellipse cx="1250" cy="130" rx="210" ry="45"/>
 <ellipse cx="1170" cy="110" rx="110" ry="55"/>
</g>
<!-- Digital globe -->
<circle cx="1080" cy="330" r="205" fill="url(#globe)"/>
<circle cx="1080" cy="330" r="205" fill="none"
 stroke="#f4fffc" stroke-width="4" opacity=".8"/>
<g fill="none" stroke="#f4fffc" stroke-width="3" opacity=".8">
 <ellipse cx="1080" cy="330" rx="85" ry="205"/>
 <ellipse cx="1080" cy="330" rx="155" ry="205"/>
 <ellipse cx="1080" cy="330" rx="205" ry="70"/>
 <ellipse cx="1080" cy="330" rx="205" ry="135"/>
 <path d="M875 330H1285 M1080 125V535"/>
</g>
<g fill="#fff">
 <circle cx="975" cy="220" r="7"/>
 <circle cx="1185" cy="275" r="8"/>
 <circle cx="1030" cy="410" r="7"/>
 <circle cx="1150" cy="430" r="6"/>
 <circle cx="1000" cy="320" r="5"/>
</g>
<g stroke="#f9ffff" stroke-width="3" opacity=".85">
 <path d="M975 220L1080 330L1185 275L1150 430L1030 410L975 220"/>
 <path d="M1000 320L1185 275 M1080 330L1150 430"/>
</g>
<!-- Wind turbines -->
<g stroke="#fff" stroke-width="7" stroke-linecap="round" opacity=".9">
 <path d="M320 390V650 M320 390L275 325 M320 390L390 350 M320 390L320 465"/>
 <path d="M500 430V650 M500 430L465 365 M500 430L565 405 M500 430L500 495"/>
 <path d="M1390 380V620 M1390 380L1340 320 M1390 380L1455 345 M1390 380L1395 450"/>
</g>
<g fill="#fff" opacity=".8">
 <circle cx="320" cy="390" r="10"/>
 <circle cx="500" cy="430" r="10"/>
 <circle cx="1390" cy="380" r="10"/>
</g>
<!-- Solar panels -->
<g stroke="#f3ffff" stroke-width="3">
 <path d="M690 600L1000 600L1040 700L665 700Z"
 fill="#217f89" opacity=".85"/>
 <path d="M700 625H1010 M690 650H1020 M680 675H1030
 M760 600L750 700 M830 600L820 700 M900 600L890 700
 M970 600L960 700" fill="none"/>
</g>
<!-- Rolling hills -->
<path d="M0 620 Q250 480 500 640 T1000 610 T1600 570 V1000H0Z"
 fill="#8dcba5"/>
<path d="M0 710 Q250 590 540 710 T1050 675 T1600 640 V1000H0Z"
 fill="url(#hill)"/>
<path d="M0 820 Q350 700 680 820 T1250 790 T1600 750 V1000H0Z"
 fill="#18775d" opacity=".65"/>
<!-- Leaves -->
<g fill="#23866a" opacity=".8">
 <path d="M95 580Q15 480 90 425Q160 475 95 580Z"/>
 <path d="M105 590Q105 490 195 480Q200 555 105 590Z"/>
 <path d="M1460 790Q1380 700 1450 650Q1525 705 1460 790Z"/>
 <path d="M1470 800Q1480 710 1560 710Q1555 780 1470 800Z"/>
</g>
<g stroke="#eafff4" stroke-width="3" fill="none" opacity=".7">
 <path d="M45 700Q100 570 95 455 M95 580L55 525 M95 550L140 505"/>
 <path d="M1420 850Q1465 750 1460 680 M1460 790L1420 745 M1460 770L1515 735"/>
</g>
</svg>
"""

svg_data = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

st.markdown(f"""
<style>
.stApp {{
    background-color: #dff2ef;
    background-image:
        linear-gradient(
            rgba(235, 250, 246, .20),
            rgba(231, 247, 240, .25)
        ),
        url("data:image/svg+xml;base64,{svg_data}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #153d4b;
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: transparent;
}}

[data-testid="stHeader"] {{
    background: rgba(235, 249, 245, .55);
}}

.block-container {{
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}}

.rag-hero {{
    padding: 1.5rem 1.8rem;
    border-radius: 22px;
    margin-bottom: 1.2rem;
    background: rgba(255, 255, 255, .83);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(29, 125, 106, .2);
    box-shadow: 0 8px 28px rgba(24, 78, 84, .10);
}}

.rag-hero h1 {{
    color: #075b62;
    font-size: 2.2rem;
}}

.rag-hero p {{
    color: #426775;
}}

section[data-testid="stSidebar"] {{
    background: rgba(244, 253, 248, .94);
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(255, 255, 255, .84);
    backdrop-filter: blur(8px);
    border-radius: 16px;
}}

[data-testid="stFileUploader"] {{
    background: rgba(255, 255, 255, .85);
    padding: 1rem;
    border-radius: 14px;
}}

.stButton > button[kind="primary"] {{
    background: #087f78;
    color: white;
    border: 0;
    border-radius: 999px;
}}

.stButton > button[kind="primary"]:hover {{
    background: #066c66;
}}

</style>
""", unsafe_allow_html=True)


# ---------------- CHAT HISTORY ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown("## 🌿 RAG Assistant")
    st.caption("Multi-Document Question Answering")
    st.divider()
    st.markdown("### 🕘 Chat History")

    if not st.session_state.chat_history:
        st.caption("Your questions and answers will appear here.")
    else:
        for item in reversed(st.session_state.chat_history):
            with st.expander(item["question"][:42]):
                st.caption(item["timestamp"])
                st.markdown("**Answer**")
                st.write(item["answer"])

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Powered by semantic search + Gemini")


# ---------------- MAIN PAGE ----------------

st.markdown("""
<div class="rag-hero">
    <h1>🌱 RAG Chat Assistant</h1>
    <p>
        Ask questions about your documents. Retrieve relevant passages
        and generate source-backed answers using Retrieval-Augmented Generation.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Upload one or more text-based PDFs. "
    "The assistant will use your documents to answer questions."
)

uploaded_files = st.file_uploader(
    "📄 Upload your PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Scanned PDFs without selectable text may not be readable."
)

all_chunks = []
chunk_sources = []

if uploaded_files:
    with st.spinner("Reading your PDF documents..."):
        for uploaded_file in uploaded_files:
            try:
                text = extract_pdf_text(uploaded_file)
                chunks = split_text(text)
                all_chunks.extend(chunks)
                chunk_sources.extend([uploaded_file.name] * len(chunks))
            except Exception as error:
                st.warning(f"Could not read {uploaded_file.name}: {error}")

    st.success(
        f"Loaded {len(uploaded_files)} PDF file(s) and "
        f"created {len(all_chunks)} text chunk(s)."
    )

    with st.expander("📚 Uploaded document details"):
        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    if not all_chunks:
        st.warning(
            "No readable text was found. Check whether your PDFs "
            "contain selectable text."
        )

    question = st.text_input(
        "💬 Ask a question about your documents",
        placeholder="Example: What is an embedding?",
        key="current_question"
    )

    if st.button("Generate Answer", type="primary",
                 use_container_width=True):

        if not question.strip():
            st.warning("Please enter a question.")

        elif not all_chunks:
            st.warning("No document text is available to search.")

        elif not os.getenv("GEMINI_API_KEY"):
            st.error("API key not found. Please check your .env file.")

        else:
            answer = None
            sources = []
            rejected = False

            try:
                with st.spinner(
                    "Searching documents and generating your answer..."
                ):
                    chunk_vectors = create_embeddings(tuple(all_chunks))
                    model = load_embedding_model()

                    question_vector = model.encode(
                        [question],
                        normalize_embeddings=True
                    )[0]

                    scores = np.dot(chunk_vectors, question_vector)
                    result_count = min(3, len(all_chunks))
                    top_indices = np.argsort(scores)[::-1][:result_count]

                    threshold = 0.25
                    best_score = float(scores[top_indices[0]])

                    if best_score < threshold:
                        answer = (
                            "The uploaded documents do not appear to contain "
                            "enough relevant information to answer this question. "
                            "Please ask a question related to your PDFs."
                        )
                        rejected = True

                    else:
                        context_parts = []

                        for rank, index in enumerate(top_indices, start=1):
                            context_parts.append(
                                f"[Source {rank}: {chunk_sources[index]}]\n"
                                f"{all_chunks[index]}"
                            )

                        context = "\n\n".join(context_parts)

                        client = genai.Client(
                            api_key=os.getenv("GEMINI_API_KEY")
                        )

                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=(
                                "You are a document question-answering assistant. "
                                "Answer using only the supplied document passages. "
                                "If the passages do not contain enough information, "
                                "say so. Do not invent facts. Mention source "
                                "filenames when possible.\n\n"
                                f"DOCUMENT PASSAGES:\n{context}\n\n"
                                f"QUESTION: {question}"
                            )
                        )

                        answer = getattr(response, "text", None)

                        if not answer or not answer.strip():
                            answer = (
                                "Gemini returned an empty answer. Please try again."
                            )
                        else:
                            sources = [
                                {
                                    "rank": rank,
                                    "filename": chunk_sources[index],
                                    "score": float(scores[index]),
                                    "text": all_chunks[index]
                                }
                                for rank, index in enumerate(
                                    top_indices, start=1
                                )
                            ]

                if answer:
                    st.session_state.chat_history.append({
                        "question": question.strip(),
                        "answer": answer,
                        "sources": sources,
                        "rejected": rejected,
                        "timestamp": datetime.now().strftime(
                            "%b %d, %I:%M %p"
                        )
                    })

            except Exception as error:
                error_message = str(error)
                error_upper = error_message.upper()

                if "503" in error_message or "UNAVAILABLE" in error_upper:
                    st.warning(
                        "Gemini is temporarily busy. Please wait and retry."
                    )
                elif (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_upper
                ):
                    st.warning(
                        "The API request limit was reached. Please wait and retry."
                    )
                elif (
                    "401" in error_message
                    or "403" in error_message
                    or "API_KEY_INVALID" in error_upper
                ):
                    st.error(
                        "Check your Gemini API key in the .env file."
                    )
                else:
                    st.error("Something went wrong.")
                    st.code(error_message)


# ---------------- CONVERSATION ----------------

st.markdown("### 💬 Conversation")

if not st.session_state.chat_history:
    st.markdown("""
    <div class="rag-hero">
        <p>
            👋 Hello! Upload your PDFs, then ask a question.
            Your conversation will appear here.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    for item in st.session_state.chat_history:
        with st.container(border=True):
            st.markdown(f"**🧑 You · {item['timestamp']}**")
            st.write(item["question"])

            st.markdown("**🌿 RAG Assistant**")
            st.write(item["answer"])

            if item.get("rejected"):
                st.caption(
                    "The question was rejected by the initial similarity "
                    "threshold; Gemini was not called."
                )

            if item.get("sources"):
                with st.expander(
                    f"📄 Retrieved sources ({len(item['sources'])})"
                ):
                    for source in item["sources"]:
                        st.markdown(
                            f"**Source {source['rank']}: "
                            f"{source['filename']}**"
                        )
                        st.caption(
                            f"Similarity score: {source['score']:.3f}"
                        )
                        st.write(source["text"])
                        st.divider()
