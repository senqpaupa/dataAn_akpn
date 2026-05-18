#!/usr/bin/env python3
"""Classify customer reviews with an LLM API and save structured JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LLM_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


JSON_SCHEMA: dict[str, Any] = {
    "name": "review_classification_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                        },
                        "topic": {
                            "type": "string",
                            "description": "Main review topic in one short English label.",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One short Russian explanation of the classification.",
                        },
                        "confidence": {
                            "type": "number",
                        },
                    },
                    "required": [
                        "id",
                        "sentiment",
                        "topic",
                        "summary",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["items"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read reviews from CSV, classify them with an LLM API, and save JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with columns id and review.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the result JSON will be saved.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model name. Default: env LLM_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "Chat Completions API URL. Default: env LLM_BASE_URL or "
            f"{DEFAULT_LLM_API_URL}."
        ),
    )
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "prompt_only"],
        default=None,
        help=(
            "How to ask the LLM for JSON. Use prompt_only if a provider does not "
            "support OpenAI-style response_format."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["llm-api", "openai", "codex-cli"],
        default="llm-api",
        help=(
            "LLM client. Use llm-api for an OpenAI-compatible HTTP API. "
            "openai is kept as an alias for llm-api. "
            "codex-cli is a local demo fallback for this repository environment."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to .env file with LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: empty key")
        os.environ.setdefault(key, value)


def read_reviews(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = {"id", "review"} - set(reader.fieldnames or [])
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required columns: {joined}")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            review_id = (row.get("id") or "").strip()
            review_text = (row.get("review") or "").strip()
            if not review_id or not review_text:
                raise ValueError(f"Row {row_number} must contain non-empty id and review")
            rows.append({"id": review_id, "review": review_text})

    if not rows:
        raise ValueError("Input CSV does not contain reviews")
    return rows


def build_messages(reviews: list[dict[str, str]]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a data-analysis assistant. Classify Russian customer reviews. "
        "Return only valid JSON that matches the provided schema. "
        "Use sentiment values exactly as positive, negative, or neutral. "
        "Use concise English topic labels such as delivery, payment, support, "
        "product_quality, price, usability, assortment, or replacement."
    )
    user_prompt = (
        "Classify each review by sentiment and main topic. "
        "For summary, write one short sentence in Russian explaining the decision.\n\n"
        f"Reviews:\n{json.dumps(reviews, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm_api(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    api_url: str,
    response_format: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        }
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed with HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM API request failed: {error.reason}") from error

    api_response = json.loads(raw_body)
    content = api_response["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            matches.append(parsed)

    if not matches:
        raise ValueError("Could not find a JSON object with an items array in LLM output")
    return matches[-1]


def call_codex_cli(reviews: list[dict[str, str]], model: str) -> dict[str, Any]:
    prompt = (
        "Classify Russian customer reviews. Return only JSON with this exact shape: "
        '{"items":[{"id":string,"sentiment":"positive"|"negative"|"neutral",'
        '"topic":string,"summary":string,"confidence":number}]}. '
        "Classify every review. Use concise English topic labels and Russian summaries. "
        "Do not use markdown.\n\n"
        f"Reviews:\n{json.dumps(reviews, ensure_ascii=False, indent=2)}"
    )
    command = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "-m",
        model,
        "-C",
        str(Path.cwd()),
        prompt,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    if completed.returncode != 0:
        raise RuntimeError(f"codex-cli provider failed:\n{combined_output}")
    return extract_json_object(combined_output)


def validate_llm_result(result: dict[str, Any], expected_ids: set[str]) -> list[dict[str, Any]]:
    items = result.get("items")
    if not isinstance(items, list):
        raise ValueError("LLM response must contain an items array")

    seen_ids: set[str] = set()
    validated_items: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} is not an object")

        review_id = str(item.get("id", "")).strip()
        sentiment = item.get("sentiment")
        topic = item.get("topic")
        summary = item.get("summary")
        confidence = item.get("confidence")

        if review_id not in expected_ids:
            raise ValueError(f"Unexpected or missing review id in item {index}: {review_id!r}")
        if review_id in seen_ids:
            raise ValueError(f"Duplicate review id in LLM response: {review_id}")
        if sentiment not in ALLOWED_SENTIMENTS:
            raise ValueError(f"Invalid sentiment for review {review_id}: {sentiment!r}")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"Invalid topic for review {review_id}")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"Invalid summary for review {review_id}")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid confidence for review {review_id}: {confidence!r}")

        seen_ids.add(review_id)
        validated_items.append(
            {
                "id": review_id,
                "sentiment": sentiment,
                "topic": topic.strip(),
                "summary": summary.strip(),
                "confidence": round(float(confidence), 3),
            }
        )

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        joined = ", ".join(sorted(missing_ids))
        raise ValueError(f"LLM response is missing review ids: {joined}")

    return sorted(validated_items, key=lambda row: int(row["id"]) if row["id"].isdigit() else row["id"])


def save_result(
    output_path: Path,
    input_path: Path,
    client: str,
    llm_provider: str,
    api_url: str | None,
    model: str,
    results: list[dict[str, Any]],
) -> None:
    output = {
        "metadata": {
            "task": "review_sentiment_and_topic_classification",
            "llm_provider": llm_provider,
            "client": client,
            "api_url": api_url,
            "model": model,
            "input_file": str(input_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "item_count": len(results),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    args.model = args.model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    args.api_url = args.api_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_LLM_API_URL
    args.response_format = args.response_format or os.environ.get("LLM_RESPONSE_FORMAT", "json_schema")
    if args.response_format not in {"json_schema", "json_object", "prompt_only"}:
        print(
            "Error: LLM_RESPONSE_FORMAT must be json_schema, json_object, or prompt_only.",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)

    reviews = read_reviews(input_path)
    if args.provider in {"llm-api", "openai"}:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: LLM_API_KEY environment variable is not set.", file=sys.stderr)
            return 2
        messages = build_messages(reviews)
        raw_result = call_llm_api(
            messages,
            args.model,
            api_key,
            args.api_url,
            args.response_format,
        )
        llm_provider = os.environ.get("LLM_PROVIDER_NAME", "openai-compatible")
        api_url = args.api_url
    else:
        raw_result = call_codex_cli(reviews, args.model)
        llm_provider = "openai"
        api_url = None

    validated_results = validate_llm_result(
        raw_result,
        expected_ids={review["id"] for review in reviews},
    )
    save_result(
        output_path,
        input_path,
        args.provider,
        llm_provider,
        api_url,
        args.model,
        validated_results,
    )

    print(f"Saved {len(validated_results)} classified reviews to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
