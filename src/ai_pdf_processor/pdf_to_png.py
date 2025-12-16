"""
Utilities to convert PDF pages to PNG images using PyMuPDF (pymupdf).

This module provides a simple function `pdf_to_png_pages` that renders all
pages of a PDF into PNG files and returns their paths.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover - import-time error surfaced at use-site
    fitz = None  # type: ignore


def pdf_to_png_pages(pdf_path: str, out_dir: Optional[str] = None, dpi: int = 144) -> List[str]:
    """
    Convert all pages of the given PDF to PNG images.

    Parameters:
        pdf_path: Path to the input PDF (must be a local file).
        out_dir: Directory to place output PNGs. If None, uses the PDF's directory.
        dpi: Rendering DPI (affects output resolution). Defaults to 144 (~2x 72dpi).

    Returns:
        A list of file paths to the generated PNG images, ordered by page number.
    """
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is required for PDF conversion. Please install 'pymupdf'."
        )

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = out_dir or os.path.dirname(os.path.abspath(pdf_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)  # scale matrix based on DPI

    png_paths: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            out_path = os.path.join(out_dir, f"{base}_page{page_index + 1}.png")
            pix.save(out_path)
            png_paths.append(out_path)

    return png_paths
