import json
import logging
import os
import socket
import sys
from urllib.parse import urlparse

import pytest


def _ensure_imports():
    try:
        import ai_pdf_processor  # noqa: F401
        return
    except Exception:
        pass
    # Add src to sys.path if package is not installed
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_imports()


from ai_pdf_processor import ask_pdf_question


def _endpoint_reachable(endpoint: str) -> bool:
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or (11434 if (parsed.scheme in ("http", "https")) else 11434)
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False


def _get_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "gemma3n:e2b")


@pytest.mark.integration
def test_basic_understanding():
    endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    logging.info(f"Using Ollama endpoint: {endpoint}")
    if not _endpoint_reachable(endpoint):
        pytest.skip(f"Ollama endpoint not reachable at {endpoint}")

    # Use sample doc from repository
    test_root = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(test_root, "Netvalue Christmas Newssheet.pdf")
    if not os.path.isfile(doc_path):
        pytest.skip("Sample doc not found in repository")

    result = ask_pdf_question(
        pdf_path=doc_path,
        # question='Analyse the provided image of the form. '
        #          'List all sections and groups of questions in the form. '
        #          'List all answer options for each question. '
        #          'Discover what are the answers to the questions, which checkboxes are checked, etc... '
        #          'Provide as many details as possible.',
        question="What image do you see?",
        model=_get_model(),
        endpoint=endpoint,
        # options={"temperature": 0},
    )

    logging.info(f"Result:\n{result}")
