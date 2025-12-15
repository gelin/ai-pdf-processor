import base64
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from ollama import Client


DEFAULT_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _load_image_as_base64(image: str) -> str:
    if _is_url(image):
        # Lightweight URL fetch without adding requests dependency
        # Use urllib to avoid external deps.
        from urllib.request import urlopen

        with urlopen(image, timeout=30) as resp:  # nosec - caller controls URL intentionally
            data = resp.read()
    else:
        with open(image, "rb") as f:
            data = f.read()
    return base64.b64encode(data).decode("ascii")


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

        b64 = _load_image_as_base64(image)
        client = Client(host=self.endpoint)
        resp = client.generate(
            model=self.model,
            prompt=question,
            images=[b64],
            options=options or None,
            stream=False,
        )
        response = (resp or {}).get("response") or {}
        if not isinstance(response, str):
            raise RuntimeError("Unexpected response from Ollama client: missing 'response'")
        return response


def ask_image_question(image: str, question: str, *, model: str, endpoint: Optional[str] = None,
                       options: Optional[Dict[str, Any]] = None, timeout: int = 120) -> str:
    client = OllamaVisionClient(endpoint=endpoint or DEFAULT_ENDPOINT, model=model, timeout=timeout)
    return client.ask(image, question, options=options)
