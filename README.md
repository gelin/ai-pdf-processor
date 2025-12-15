# AI PDF Processor

A tiny Python library and CLI that talk to a local Ollama instance to answer
questions about an image using a vision model (e.g., `llava`). This is a
building block for processing PDF scans or images and extracting answers.

Input: image (or PDF converted to images) + questions.
Output: textual answers (you can interpret them as boolean, number, or string in your app).

Uses a small local LLaVA model via Ollama to analyze images.

## Requirements

- Python 3.9+
- Install the official client: `pip install ollama`
- A running Ollama server (default: `http://localhost:11434`)
- A vision-capable model pulled, e.g.: `ollama pull llava:7b`

You can override the endpoint with `OLLAMA_ENDPOINT` env var.

## Library Usage

```
pip install ollama

from ai_pdf_processor import ask_image_question

answer = ask_image_question(
    image="/path/to/image.png",  # or a URL
    question="What is written in the top-right box?",
    model="llava:7b",            # required
    endpoint="http://localhost:11434",  # optional (can use $OLLAMA_ENDPOINT)
    options={"temperature": 0}, # optional
)
print(answer)
```

## CLI Usage

You can run the CLI module directly:

```
python -m ai_pdf_processor.cli /path/to/image.png "What is the invoice total?"
```

Options:

```
python -m ai_pdf_processor.cli \
  --model llava:7b \
  --endpoint http://localhost:11434 \
  --option temperature=0 \
  --json \
  https://example.com/form.jpg "Is the checkbox marked?"
```

Environment variables:
- `OLLAMA_ENDPOINT` – default Ollama HTTP endpoint

CLI exits with non-zero on error. Use `--json` to get `{ "answer": "..." }`.

## Notes

- If you work with PDFs, convert pages to images first (e.g., with `pdftoppm` or `pypdfium2`).
- For structured outputs (bool/number/string), add instructions in the question, e.g.,
  "Answer with only true/false" or "Respond with only a number".

## Tags

REST API. Python lib to Ollama. Conversion of PDF to PNG.

## Models

These models can be good for this task:
* [llava:7b](https://ollama.com/library/llava)
* [gemma3n:e2b](https://ollama.com/library/gemma3n)
