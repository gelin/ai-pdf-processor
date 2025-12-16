"""
AI PDF Processor - Ollama vision helper library

This package provides a tiny wrapper around the Ollama API (via the official
ollama-python client) to ask questions about an image using a vision-capable
model (e.g., llava).
"""

from .ollama_vision import ask_image_question, OllamaVisionClient
from .pdf_to_png import pdf_to_png_pages


def ask_pdf_question(pdf_path: str, question: str, *, page: int = 1, model: str,
                     endpoint: str | None = None, options=None, timeout: int = 120) -> str:
    """Convert the selected PDF page to PNG and ask the question.

    Parameters:
        pdf_path: Local path to the PDF file.
        question: The user's question about the page.
        page: 1-based page number to analyze.
        model, endpoint, options, timeout: forwarded to Ollama.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ai_pdf_proc_") as tmpdir:
        pngs = pdf_to_png_pages(pdf_path, out_dir=tmpdir)
        if not pngs:
            raise RuntimeError("No pages found in PDF")
        idx = page - 1
        if idx >= len(pngs):
            raise ValueError(f"Requested page {page} exceeds total pages {len(pngs)}")
        return ask_image_question(
            pngs[idx],
            question,
            model=model,
            endpoint=endpoint,
            options=options,
            timeout=timeout,
        )

__all__ = [
    "OllamaVisionClient",
    "ask_image_question",
    "ask_pdf_question",
    "pdf_to_png_pages",
]
