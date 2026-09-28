from __future__ import annotations

import streamlit as st

from src.auth import require_password
from src.rag_pipeline import ask

st.set_page_config(page_title="AgriRAG", page_icon="🌿", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
:root { --agri-green: #1f5f3b; --sand: #f6f1e7; }
html, body, [class*="css"] { font-family: Inter, Arial, sans-serif; }
.rtl, .rtl * { direction: rtl; text-align: right; font-family: 'Amiri', serif !important; }
.hero { padding: 1rem 1.2rem; border-radius: 18px; background: linear-gradient(135deg, #eef7f1, #faf7ef); border: 1px solid #dfe9e1; }
.source-card { border: 1px solid #e5e5e5; border-radius: 12px; padding: .75rem; margin-bottom: .5rem; }
</style>
""",
    unsafe_allow_html=True,
)

if not require_password():
    st.stop()

with st.sidebar:
    language = st.radio("Interface / الواجهة", ["العربية", "English"], horizontal=True)
    top_n = st.slider("Sources / المصادر", 3, 8, 5)
    st.caption("Hybrid retrieval: vector + BM25 + RRF, then multilingual reranking.")

rtl = language == "العربية"
if rtl:
    st.markdown('<div class="rtl">', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🌿 AgriRAG — مساعد المعرفة الزراعية</h1><p>إجابات مستندة إلى المصادر في الري، التربة، تغذية النبات والزراعة الذكية مناخياً.</p></div>', unsafe_allow_html=True)
    question = st.text_area("سؤالك", placeholder="مثال: ما العوامل التي تؤثر في جدولة الري؟")
    ask_label = "اسأل"
else:
    st.markdown('<div class="hero"><h1>🌿 AgriRAG — Evidence-Grounded Agriculture Assistant</h1><p>Source-grounded answers on irrigation, soils, plant nutrition, and climate-smart agriculture.</p></div>', unsafe_allow_html=True)
    question = st.text_area("Your question", placeholder="Example: Which factors should irrigation scheduling consider?")
    ask_label = "Ask"

if st.button(ask_label, type="primary") and question.strip():
    with st.spinner("Searching and reranking..." if not rtl else "جاري البحث وإعادة الترتيب..."):
        try:
            result = ask(question.strip(), top_n=top_n)
        except Exception as exc:
            st.exception(exc)
        else:
            st.markdown(result["answer"])
            st.caption(f"Latency: {result['latency_seconds']:.2f}s")
            with st.expander("Sources / المصادر", expanded=True):
                for i, source in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {source['source']} — page {source.get('page') or '?'}**  \n"
                        f"{source['text'][:900]}{'…' if len(source['text']) > 900 else ''}"
                    )

if rtl:
    st.markdown('</div>', unsafe_allow_html=True)
