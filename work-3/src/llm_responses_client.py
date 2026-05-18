"""Minimal Responses API client for the Streamlit data analyst app."""

from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class GeneratedFile:
    file_id: str
    container_id: str
    filename: str
    content: bytes | None = None


@dataclass(frozen=True)
class AnalysisResult:
    report_text: str
    raw_response: dict[str, Any]
    generated_files: list[GeneratedFile]
    code_calls: list[dict[str, Any]]


class ResponsesClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def upload_file(self, path: Path, purpose: str = "user_data") -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "text/csv"
        with path.open("rb") as file:
            response = self.session.post(
                f"{self.base_url}/files",
                data={"purpose": purpose},
                files={"file": (path.name, file, mime_type)},
                timeout=120,
            )
        self._raise_for_status(response)
        payload = response.json()
        return payload["id"]

    def analyze_csv(
        self,
        *,
        model: str,
        file_id: str,
        instructions: str,
        user_prompt: str,
        memory_limit: str,
    ) -> AnalysisResult:
        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user_prompt,
            "tools": [
                {
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",
                        "memory_limit": memory_limit,
                        "file_ids": [file_id],
                    },
                }
            ],
            "tool_choice": "required",
        }

        response = self.session.post(
            f"{self.base_url}/responses",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        self._raise_for_status(response)
        raw_response = response.json()
        report_text = extract_output_text(raw_response)
        code_calls = extract_code_calls(raw_response)
        files = self._download_generated_files(extract_generated_file_refs(raw_response))
        return AnalysisResult(
            report_text=report_text,
            raw_response=raw_response,
            generated_files=files,
            code_calls=code_calls,
        )

    def _download_generated_files(self, refs: list[GeneratedFile]) -> list[GeneratedFile]:
        downloaded: list[GeneratedFile] = []
        seen: set[tuple[str, str]] = set()
        for ref in refs:
            key = (ref.container_id, ref.file_id)
            if key in seen:
                continue
            seen.add(key)
            response = self.session.get(
                f"{self.base_url}/containers/{ref.container_id}/files/{ref.file_id}/content",
                timeout=120,
            )
            if response.ok:
                downloaded.append(
                    GeneratedFile(
                        file_id=ref.file_id,
                        container_id=ref.container_id,
                        filename=safe_filename(ref.filename or ref.file_id),
                        content=response.content,
                    )
                )
            else:
                downloaded.append(ref)
        return downloaded

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"API request failed with HTTP {response.status_code}: {detail}"
        )


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()

    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n\n".join(chunks).strip()


def extract_code_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output", [])
        if item.get("type") == "code_interpreter_call"
    ]


def extract_generated_file_refs(response: dict[str, Any]) -> list[GeneratedFile]:
    refs: list[GeneratedFile] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            for annotation in content.get("annotations", []):
                if annotation.get("type") != "container_file_citation":
                    continue
                file_id = annotation.get("file_id")
                container_id = annotation.get("container_id")
                if not file_id or not container_id:
                    continue
                refs.append(
                    GeneratedFile(
                        file_id=file_id,
                        container_id=container_id,
                        filename=annotation.get("filename") or file_id,
                    )
                )
    return refs


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "generated_file"


def response_to_pretty_json(response: dict[str, Any]) -> str:
    return json.dumps(response, ensure_ascii=False, indent=2)
