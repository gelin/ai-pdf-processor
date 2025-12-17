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
        description="Ask one or multiple typed questions about a local image or a PDF page using an Ollama vision model.",
    )
    p.add_argument("document", help="Local path to the image or PDF (URLs are not allowed)")
    p.add_argument("question", nargs="?", help="Single question to ask (fallback if --questions/--questions-file not used)")
    p.add_argument("--model", required=True,
                   help="Ollama model to use (required), e.g. llava:7b")
    p.add_argument("--endpoint", default=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434"),
                   help="Ollama HTTP endpoint (default: http://localhost:11434 or $OLLAMA_ENDPOINT)")
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="Model generation option (repeatable), e.g. --option temperature=0 --option num_ctx=4096")
    p.add_argument("--json", action="store_true", help="Output JSON (for single question, wraps as {answer: ...})")
    p.add_argument(
        "--page",
        type=int,
        default=1,
        help="When input is a PDF, which page to use (1-based index, default: 1)",
    )
    qgrp = p.add_argument_group("Multiple questions")
    qgrp.add_argument(
        "--questions",
        help=(
            "JSON array of questions, e.g. "
            "'[{""question"": ""Total sum?"", ""type"": ""number""}, {""question"": ""Has signature?"", ""type"": ""boolean""}]'"
        ),
    )
    qgrp.add_argument(
        "--questions-file",
        help="Path to a JSON file containing the questions array",
    )
    qgrp.add_argument(
        "--type",
        default="string",
        choices=["string", "number", "boolean"],
        help="Expected type for the single positional -- question (default: string)",
    )
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
        inp = args.document
        # Determine questions list
        q_list = None
        if args.questions_file:
            with open(args.questions_file, "r", encoding="utf-8") as f:
                q_list = json.load(f)
        elif args.questions:
            q_list = json.loads(args.questions)
        elif args.question:
            q_list = [{"question": args.question, "type": args.type}]
        else:
            raise SystemExit("Provide either positional QUESTION or --questions/--questions-file")

        # Call unified API
        try:
            from . import ask_document_questions
        except Exception:
            from ai_pdf_processor import ask_document_questions

        if args.page < 1:
            raise SystemExit("--page must be >= 1")

        result = ask_document_questions(
            path=inp,
            questions=q_list,
            page=args.page,
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

    # Output
    if isinstance(result, dict) and "answers" in result:
        print(json.dumps(result, ensure_ascii=False))
    else:
        # Fallback for backward compatibility when using single-question path
        if args.json:
            print(json.dumps({"answer": result}, ensure_ascii=False))
        else:
            print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
