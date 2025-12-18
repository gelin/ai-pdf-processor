import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal

from ollama import Client


DEFAULT_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _load_local_image_as_base64(image_path: str) -> str:
    if _is_url(image_path):
        raise ValueError("Only local files are supported; URLs are not allowed.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        data = f.read()
    logging.debug(f"Loaded image from {image_path} ({len(data)} bytes)")
    b64 = base64.b64encode(data).decode("ascii")
    logging.debug(f"Converted to base64: {b64[:10]}...")
    return b64


@dataclass(frozen=True)
class Question:
    question: str
    type: Literal["string", "number", "boolean"] = "string"


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answers"],
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question", "answer"],
                "properties": {
                    "question": {
                        "type": "string",
                    },
                    "answer": {
                        "type": ["string", "number", "boolean", "null"]
                    },
                    "comment": {
                        "type": "string"
                    }
                }
            }
        }
    }
}


def _build_batch_prompt(questions: List[Question]) -> str:
    lines: List[str] = []
    lines.append(
        "You're analysing one page or a scanned document image. Carefully read all visible text and layout."
    )
    lines.append(
        'Answer the following questions. Return a STRICT JSON object with the schema:\n'
        '{\n  "answers": ['
        '{ "question": "<original question>", "answer": "<your answer with appropriate JSON type>", "comment": "<your optional comment>"}'
        '<...one answer per question, same order...>'
        ']\n}'
    )
    lines.append("Rules:")
    lines.append("- Quote the original question in 'question' field of the answer JSON object.")
    lines.append("- Put your answer in 'answer' field of the answer JSON object.")
    lines.append("- In answer value use native JSON types only: string, number, boolean, or null when uncertain.")
    lines.append("- Add optional 'comment' field with your comments to the answer JSON object if needed.")
    lines.append("- Do not include any extra keys or text before/after the JSON.")
    lines.append("- The length of 'answers' must equal the number of questions asked.")
    lines.append("")
    lines.append("Questions (with expected answer types):")
    for i, q in enumerate(questions, start=1):
        qtext = (q.question or "").strip()
        qtype = (q.type or "string").strip().lower()
        lines.append(f"{i}. {qtext} (answer with {qtype} JSON type)")
    lines.append("")
    lines.append("Output only:")
    lines.append('{\n  \"answers\": [ { "question": ..., "answer": ..., "comment": ... }, ... ]\n}')
    return "\n".join(lines)


@dataclass
class OllamaVisionClient:
    endpoint: str = DEFAULT_ENDPOINT
    model: Optional[str] = None
    timeout: int = 120

    def ask(self, image: str, question: str, *, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a question about the provided image using a vision-capable model.

        Parameters:
            image: Local file path or URL to the image.
            question: The user's question.
            options: Optional dict with model generation options (e.g., {"temperature": 0}).

        Returns:
            The assistant's textual answer.
        """
        if not self.model:
            raise ValueError("Model must be provided (no default). Set OllamaVisionClient.model or pass model to ask_image_question().")

        b64 = _load_local_image_as_base64(image)
        client = Client(host=self.endpoint)
        logging.debug(f"Prompt:\n{question}")
        logging.debug(f"Image:\n{b64[:10]}...")
        resp = client.generate(
            model=self.model,
            prompt=question,
            images=[b64],
            options=options or None,
            # format=RESPONSE_SCHEMA,
            stream=False,
        )
        response = (resp or {}).get("response") or {}
        if not isinstance(response, str):
            raise RuntimeError("Unexpected response from Ollama client: missing 'response'")
        return response

    def ask_many(self, image: str, questions: List[Question], *, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ask multiple questions about the provided image in a single call.

        Parameters:
            image: Local file path to the image.
            questions: List of {"question": str, "type": "string"|"number"|"boolean"}.
            options: Optional dict with model generation options.

        Returns:
            Parsed JSON dict from the model, expected to contain {"answers": [...]}
        """
        if not self.model:
            raise ValueError("Model must be provided (no default). Set OllamaVisionClient.model.")

        if not isinstance(questions, list) or not questions:
            raise ValueError("'questions' must be a non-empty list")
        # Basic validation of types for dataclass Question
        allowed = {"string", "number", "boolean"}
        for q in questions:
            if not isinstance(q, Question):
                raise ValueError("Each item in 'questions' must be a Question dataclass instance")
            if (q.type or "string").lower() not in allowed:
                raise ValueError(f"Unsupported question type '{q.type}'. Allowed: string, number, boolean")

        prompt = _build_batch_prompt(questions)
        logging.info(f"Generated prompt:\n{prompt}")

        b64 = _load_local_image_as_base64(image)
        logging.debug(f"Image:\n{b64[:10]}...")

        client = Client(host=self.endpoint)

        # Ensure JSON mode is enabled
        merged_options = dict(options or {})
        merged_options.setdefault("format", "json")

        resp = client.generate(
            model=self.model,
            prompt=prompt,
            images=[b64],
            options=merged_options,
            format=RESPONSE_SCHEMA,
            stream=False,
        )
        response = (resp or {}).get("response")
        if not isinstance(response, str):
            raise RuntimeError("Unexpected response from Ollama client: missing 'response'")
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw: {response[:500]}")
        if not isinstance(data, dict) or "answers" not in data:
            raise RuntimeError("Model JSON missing 'answers' array")
        # if not isinstance(data["answers"], list) or len(data["answers"]) != len(questions):
        #     raise RuntimeError("'answers' must be a list with the same length as questions")
        return data


def ask_image_question(image: str, question: str, *, model: str, endpoint: Optional[str] = None,
                       options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> str:
    client = OllamaVisionClient(endpoint=endpoint or DEFAULT_ENDPOINT, model=model, timeout=timeout)
    return client.ask(image, question, options=options)


def ask_image_questions(
    image: str,
    questions: List[Question],
    *,
    model: str,
    endpoint: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Unified image entrypoint for multiple questions (JSON answers)."""
    client = OllamaVisionClient(endpoint=endpoint or DEFAULT_ENDPOINT, model=model, timeout=timeout)
    return client.ask_many(image, questions, options=options)
