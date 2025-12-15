"""
AI PDF Processor - Ollama vision helper library

This package provides a tiny wrapper around the Ollama API (via the official
ollama-python client) to ask questions about an image using a vision-capable
model (e.g., llava).
"""

from .ollama_vision import ask_image_question, OllamaVisionClient

__all__ = [
    "OllamaVisionClient",
    "ask_image_question",
]
