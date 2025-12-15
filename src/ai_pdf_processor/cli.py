#!/usr/bin/env python3
import argparse
import json
import os
import sys

try:
    from .ollama_vision import ask_image_question
except ImportError:
    # If executed directly (e.g., python src/ai_pdf_processor/cli.py),
    # __package__ is not set and relative imports fail. Add the package
    # parent directory to sys.path and use an absolute import.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG_PARENT = os.path.dirname(_HERE)
    if _PKG_PARENT not in sys.path:
        sys.path.insert(0, _PKG_PARENT)
    from ai_pdf_processor.ollama_vision import ask_image_question

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ask a question about an image using an Ollama vision model (e.g., llava).",
    )
    p.add_argument("image", help="Path or URL to the image")
    p.add_argument("question", help="Question to ask about the image")
    p.add_argument("--model", required=True,
                   help="Ollama model to use (required), e.g. llava:7b")
    p.add_argument("--endpoint", default=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434"),
                   help="Ollama HTTP endpoint (default: http://localhost:11434 or $OLLAMA_ENDPOINT)")
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="Model generation option (repeatable), e.g. --option temperature=0 --option num_ctx=4096")
    p.add_argument("--json", action="store_true", help="Output JSON with {answer: ...}")
    return p


def parse_options(option_kv_list):
    options = {}
    for item in option_kv_list:
        if "=" not in item:
            raise SystemExit(f"Invalid --option '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        # Try to cast numbers and booleans
        if v.lower() in ("true", "false"):
            v_cast = v.lower() == "true"
        else:
            try:
                if "." in v:
                    v_cast = float(v)
                else:
                    v_cast = int(v)
            except ValueError:
                v_cast = v
        options[k] = v_cast
    return options or None


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        answer = ask_image_question(
            args.image,
            args.question,
            model=args.model,
            endpoint=args.endpoint,
            options=parse_options(args.option),
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"answer": answer}, ensure_ascii=False))
    else:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
