from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from src.auth import require_password
from src.rag_pipeline import ask

st.set_page_config(
    page_title="AgriRAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOP_N = 5
ROOT = Path(__file__).resolve().parent
QUESTION_BANK = ROOT / "data" / "eval" / "golden_questions.json"
logger = logging.getLogger("agrirag")


@st.cache_data(show_spinner=False)
def load_sample_questions() -> list[str]:
    try:
        payload = json.loads(QUESTION_BANK.read_text(encoding="utf-8"))
        questions = [
            item["question"].strip()
            for item in payload.get("questions", [])
            if item.get("question")
        ]
        return questions
    except Exception:
        logger.exception("Could not load the sample-question bank")
        return [
            "How does FAO define reference evapotranspiration (ETo)?",
            "How can salinity in irrigation water reduce crop growth and yield?",
            "What is deficit irrigation?",
            "What are the three main objectives of climate-smart agriculture?",
            "How does soil organic matter improve soil structure, water management, and nutrient retention?",
        ]


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  color-scheme: light;
  --forest: #174a32;
  --forest-2: #226241;
  --leaf: #2d7a50;
  --paper: #ffffff;
  --page: #f4f8f3;
  --ink: #173126;
  --muted: #64746b;
  --line: #dbe7dd;
  --gold: #c69b4a;
}

html, body, .stApp {
  font-family: 'Inter', Arial, sans-serif;
  background: var(--page) !important;
  color: var(--ink) !important;
}

[data-testid="stHeader"] {
  background: rgba(244, 248, 243, .92) !important;
  backdrop-filter: blur(8px);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #173f2d 0%, #102f23 100%) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
  color: #f7fbf7 !important;
}

[data-testid="stSidebar"] .stCaption {
  color: #d6e5d9 !important;
}

.block-container {
  max-width: 1180px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}

.hero {
  background:
    radial-gradient(circle at 85% 18%, rgba(255,255,255,.12), transparent 24%),
    linear-gradient(135deg, #174a32 0%, #235f40 52%, #2f7650 100%);
  color: #ffffff !important;
  border-radius: 24px;
  padding: 2.2rem 2.3rem;
  box-shadow: 0 18px 48px rgba(23,74,50,.16);
  margin-bottom: .85rem;
  border: 1px solid rgba(255,255,255,.12);
}

.hero h1, .hero p, .hero span, .hero strong {
  color: #ffffff !important;
}

.hero h1 {
  margin: 0 0 .55rem 0;
  font-size: clamp(2rem, 4vw, 3.15rem);
  line-height: 1.05;
}

.hero p {
  margin: 0;
  max-width: 820px;
  font-size: 1.05rem;
  opacity: .93;
}

.pill-row {
  display: flex;
  gap: .55rem;
  flex-wrap: wrap;
  margin-top: 1.1rem;
}

.pill {
  display: inline-flex;
  align-items: center;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.18);
  color: #ffffff !important;
  border-radius: 999px;
  padding: .38rem .7rem;
  font-size: .82rem;
}

.byline {
  color: #52675a !important;
  font-size: .9rem;
  margin: .2rem 0 1rem .15rem;
}

.byline strong {
  color: var(--forest) !important;
}

.info-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: .8rem;
  margin-bottom: 1.15rem;
}

.info-card {
  background: var(--paper);
  color: var(--ink) !important;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: .9rem 1rem;
  box-shadow: 0 8px 24px rgba(24,59,40,.05);
}

.info-card strong, .info-card span {
  color: var(--ink) !important;
}

.info-card span {
  display: block;
  color: var(--muted) !important;
  font-size: .8rem;
  margin-top: .15rem;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
  background: #ffffff !important;
  color: #173126 !important;
  border-color: #cbdace !important;
}

[data-testid="stTextArea"] textarea {
  border-radius: 14px !important;
  box-shadow: 0 6px 18px rgba(28,67,44,.04);
}

[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
  color: #7a8a80 !important;
  opacity: 1 !important;
}

[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMarkdownContainer"],
.stCaption,
p, li, h1, h2, h3, h4 {
  color: var(--ink);
}

div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1f6542, #2e7a50) !important;
  color: #ffffff !important;
  border: 0 !important;
  border-radius: 12px !important;
  min-height: 44px;
  font-weight: 700;
  padding: .55rem 1.2rem;
  box-shadow: 0 8px 18px rgba(31,101,66,.18);
}

div.stButton > button[kind="primary"]:hover {
  background: #174a32 !important;
}

[data-testid="stAlert"] {
  border-radius: 14px !important;
}

[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  overflow: hidden;
}

[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
  color: var(--ink) !important;
}

.answer-heading {
  display: flex;
  align-items: center;
  gap: .55rem;
  margin: 1.4rem 0 .65rem 0;
  font-size: 1rem;
  font-weight: 700;
  color: var(--forest) !important;
}

.source-meta {
  color: var(--muted) !important;
  font-size: .82rem;
}

.small-note {
  color: var(--muted) !important;
  font-size: .84rem;
  margin-top: .35rem;
}

.footer {
  margin-top: 2.4rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
  color: var(--muted) !important;
  font-size: .84rem;
  text-align: center;
}

.footer strong {
  color: var(--forest) !important;
}

hr {
  border-color: var(--line) !important;
}

@media (max-width: 800px) {
  .block-container { padding-top: 1rem; }
  .hero { padding: 1.45rem; border-radius: 18px; }
  .info-strip { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

if not require_password():
    st.stop()

with st.sidebar:
    st.markdown("### 🌿 AgriRAG")
    st.caption("Evidence-grounded agriculture assistant")
    st.divider()
    st.markdown("**Retrieval pipeline**")
    st.caption("Vector + BM25 + RRF, followed by multilingual reranking.")
    st.markdown("**Evidence shown**")
    st.caption("Top 5 reranked passages, matching the validated evaluation setting.")
    st.divider()
    st.markdown("**Done by:** Abdullah Mofadl")

st.markdown(
    """
<div class="hero">
  <h1>🌿 AgriRAG</h1>
  <p>An evidence-grounded agriculture assistant for irrigation, soils, plant nutrition, water harvesting, and climate-smart agriculture.</p>
  <div class="pill-row">
    <span class="pill">✓ Hybrid retrieval</span>
    <span class="pill">✓ Top-5 evidence</span>
    <span class="pill">✓ Source-grounded answers</span>
  </div>
</div>
<div class="byline">Done by: <strong>Abdullah Mofadl</strong></div>
<div class="info-strip">
  <div class="info-card"><strong>90%</strong><span>Final Recall@5</span></div>
  <div class="info-card"><strong>20/20</strong><span>RAGAS questions completed</span></div>
  <div class="info-card"><strong>5</strong><span>Evidence passages per answer</span></div>
</div>
""",
    unsafe_allow_html=True,
)

sample_questions = load_sample_questions()
question_mode = st.selectbox(
    "Choose a question",
    ["Write my own question"] + sample_questions,
    index=0,
    help="The list contains the validated golden-question bank used for retrieval evaluation.",
)

if question_mode == "Write my own question":
    question = st.text_area(
        "Your question",
        placeholder="Example: How does irrigation-water salinity affect crop growth?",
        height=120,
    )
else:
    question = question_mode
    st.info(f"Selected question: {question}")

if st.button("Ask AgriRAG", type="primary") and question.strip():
    with st.spinner("Searching sources and reranking evidence..."):
        try:
            result = ask(question.strip(), top_n=TOP_N)
        except Exception:
            logger.exception("AgriRAG query failed")
            st.error(
                "The question could not be completed right now. "
                "The server error has been logged; please try again shortly."
            )
        else:
            st.markdown(
                '<div class="answer-heading">🌱 Evidence-grounded answer</div>',
                unsafe_allow_html=True,
            )
            st.markdown(result["answer"])
            st.caption(f"Answer latency: {result['latency_seconds']:.2f}s")

            with st.expander(
                f"📚 Sources and evidence used ({len(result['sources'])})",
                expanded=True,
            ):
                for i, source in enumerate(result["sources"], start=1):
                    source_name = Path(source["source"]).name
                    page = source.get("page")
                    page_text = "—" if page in (None, -1) else str(page)
                    score = source.get("rerank_score")
                    score_text = (
                        f" · relevance {score:.3f}"
                        if isinstance(score, (float, int))
                        else ""
                    )
                    st.markdown(
                        f"**[{i}] {source_name}**  \n"
                        f"<span class='source-meta'>Page {page_text}{score_text}</span>",
                        unsafe_allow_html=True,
                    )
                    excerpt = source["text"][:700]
                    if len(source["text"]) > 700:
                        excerpt += "…"
                    st.write(excerpt)
                    if i < len(result["sources"]):
                        st.divider()

            st.markdown(
                '<div class="small-note">AgriRAG shows only the top 5 reranked '
                'evidence passages to match the final validated evaluation setup.</div>',
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer">AgriRAG · Done by: <strong>Abdullah Mofadl</strong></div>',
    unsafe_allow_html=True,
)
