import json
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

from ai_pdf_processor import ask_document_questions, Question  # noqa: E402


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
    return os.environ.get("OLLAMA_MODEL", "llava:7b")


@pytest.mark.integration
def test_questions_christmas_doc_good_questions():
    endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    if not _endpoint_reachable(endpoint):
        pytest.skip(f"Ollama endpoint not reachable at {endpoint}")

    # Use sample doc from repository
    test_root = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(test_root, "Netvalue Christmas Newssheet.pdf")
    if not os.path.isfile(doc_path):
        pytest.skip("Sample doc not found in repository")

    questions = [
        Question(question="What is the title of the form?", type="string"),
        Question(question="In the 'What is your favourite Christmas song' group, is the 'All I want for Christmas is my 2 front teeth' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas song' group, is the 'Rudolf the red nosed reindeer' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas song' group, is the 'Silent night' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas song' group, is the 'Snoopys Christmas' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas treat' group, is the 'Mince pies' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas treat' group, is the 'Gingerbread men' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas treat' group, is the 'Pavlova' option checked?", type="boolean"),
        Question(question="In the 'What is your favourite Christmas treat' group, is the 'Christmas pudding with brandy butter' option checked?", type="boolean"),
        Question(question="Is the answer to 'Do you want to sign up for next years Newssheet?' question 'Yes'?", type="boolean"),
    ]
    print(questions)

    result = ask_document_questions(
        path=doc_path,
        questions=questions,
        page=1,
        model=_get_model(),
        endpoint=endpoint,
        options={"temperature": 0},
        timeout=180,
    )

    print(json.dumps(result, indent=2))
    assert isinstance(result, dict)
    assert "answers" in result and isinstance(result["answers"], list)
    assert len(result["answers"]) == len(questions)
    for item in result["answers"]:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item


@pytest.mark.integration
def test_questions_christmas_doc_hard_questions():
    endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    if not _endpoint_reachable(endpoint):
        pytest.skip(f"Ollama endpoint not reachable at {endpoint}")

    # Use sample doc from repository
    test_root = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(test_root, "Netvalue Christmas Newssheet.pdf")
    if not os.path.isfile(doc_path):
        pytest.skip("Sample doc not found in repository")

    questions = [
        Question(question="When will the office be shut down?", type="string"),     # TODO: support date type?
        Question(question="When is the hackathon presentation going to be made?", type="string"),
        Question(question="What sort of spy hides in a bakery?", type="string"),
        Question(question="How do you get frostbite?", type="string"),
        Question(question="What are the persons favourite Christmas songs?", type="string"),
        Question(question="How many ticks are on the page?", type="number"),
        Question(question="Have they signed up for next years newssheet?", type="boolean"),
    ]
    print(questions)

    result = ask_document_questions(
        path=doc_path,
        questions=questions,
        page=1,
        model=_get_model(),
        endpoint=endpoint,
        options={"temperature": 0},
        timeout=180,
    )

    print(json.dumps(result, indent=2))
    assert isinstance(result, dict)
    assert "answers" in result and isinstance(result["answers"], list)
    assert len(result["answers"]) == len(questions)
    for item in result["answers"]:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
