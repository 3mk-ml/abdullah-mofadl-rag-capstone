from src.text_utils import tokenize
from src.chunking import recursive_chunks


def test_tokenize_mixed_language():
    tokens = tokenize("الري بالتنقيط Drip-Irrigation ET0 25%")
    assert "الري" in tokens
    assert any("drip-irrigation" == t for t in tokens)


def test_chunker_returns_text():
    text = ("Agriculture water management sentence. " * 500).strip()
    chunks = recursive_chunks(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)
