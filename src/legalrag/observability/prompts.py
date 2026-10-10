"""Prompts by label: which system prompt is live is a pointer in Langfuse, not a code deploy.

    # 1. a new version enters as a candidate (never straight to production)
    uv run python -m legalrag.observability.prompts push --text-file new.txt --prompt-version v4
    # 2. the quality gate runs on exactly that text and writes a verdict with its sha256
    uv run python -m legalrag.eval.smoke --prompt-file new.txt --prompt-version v4
    # 3. promotion checks the verdict against the version's text, then moves the label
    uv run python -m legalrag.observability.prompts promote --version 2 --verdict reports/eval/smoke/verdict.json
    # rollback: only to a version that passed the gate before (no new evaluation needed)
    uv run python -m legalrag.observability.prompts rollback --version 1

Langfuse keeps every version of ``legal-rag-system``; the label ``production`` points at one.
The API asks for the version behind its label (``PROMPT_LABEL``) through the SDK's cache
(60 s per process), so a promotion or a rollback reaches every worker within about a minute,
with no build and no restart. If Langfuse cannot be reached, the prompt in the code is served
(``generation.SYSTEM_PROMPT``): that text must always be the last gated one.

The gate is in code, not only in a docstring (found in review): ``promote`` refuses unless the
verdict passed AND its ``prompt_sha256`` equals the hash of the version's text, and it records
the hash in ``reports/prompts/approved.json``; ``rollback`` only accepts recorded hashes. (The
Langfuse UI can still move labels by hand: access to it is the remaining trust boundary.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legalrag.generation import PROMPT_VERSION, SYSTEM_PROMPT
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

PROMPT_NAME = "legal-rag-system"
APPROVED = Path("reports/prompts/approved.json")


@dataclass(frozen=True)
class ServedPrompt:
    text: str
    version: str  # e.g. "v3": the evaluated prompt version, not Langfuse's counter
    source: str  # "code", "file" or "langfuse:<n>"
    client: Any = None  # the Langfuse prompt object, so traces link generations to it


class CodePrompt:
    def get(self) -> ServedPrompt:
        return ServedPrompt(SYSTEM_PROMPT, PROMPT_VERSION, "code")


class StaticPrompt:
    """A prompt from a file: what the quality gate evaluates before the text gets a label."""

    def __init__(self, text: str, version: str):
        self.text, self.version = text, version

    def get(self) -> ServedPrompt:
        return ServedPrompt(self.text, self.version, "file")


class LangfusePrompt:
    def __init__(self, client: Any, label: str = "production", cache_ttl_seconds: int = 60):
        self.client, self.label, self.ttl = client, label, cache_ttl_seconds

    def get(self) -> ServedPrompt:
        try:
            p = self.client.get_prompt(PROMPT_NAME, label=self.label, type="text",
                                       cache_ttl_seconds=self.ttl, fallback=SYSTEM_PROMPT)  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - answers must not depend on Langfuse being up
            log.warning("prompt_fetch_failed", error=type(exc).__name__)
            return CodePrompt().get()
        if getattr(p, "is_fallback", False):
            return CodePrompt().get()
        version = (p.config or {}).get("prompt_version", f"langfuse-{p.version}")
        return ServedPrompt(p.prompt, version, f"langfuse:{p.version}", p)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def move_label(client: Any, version: int, label: str) -> None:
    """Point ``label`` at ``version`` (Langfuse removes it from the version that had it)."""
    client.update_prompt(name=PROMPT_NAME, version=version, new_labels=[label])


def _version_text(client: Any, version: int) -> str:
    return client.get_prompt(PROMPT_NAME, version=version, type="text", cache_ttl_seconds=0).prompt


def _approved(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def check_push_labels(labels: list[str]) -> list[str]:
    if "production" in labels:
        raise SystemExit("a new version cannot be pushed as production: push it as a candidate, "
                         "run the quality gate on it, then promote")  # fmt: skip
    return labels or ["candidate"]


def promote(client: Any, version: int, verdict: dict[str, Any], approved_path: Path) -> None:
    text = _version_text(client, version)
    if not verdict.get("passed"):
        raise SystemExit(f"the quality gate did not pass: {verdict.get('reasons')}")
    if verdict.get("prompt_sha256") != prompt_sha256(text):
        raise SystemExit("the verdict is for a different prompt text than this version")
    approved = _approved(approved_path)
    approved[prompt_sha256(text)] = {"version": version, "approved_at": datetime.now(UTC).isoformat(),
                                     "faithfulness": verdict.get("faithfulness")}  # fmt: skip
    approved_path.parent.mkdir(parents=True, exist_ok=True)
    approved_path.write_text(json.dumps(approved, indent=2) + "\n", encoding="utf-8", newline="\n")
    move_label(client, version, "production")


def rollback(client: Any, version: int, approved_path: Path) -> None:
    if prompt_sha256(_version_text(client, version)) not in _approved(approved_path):
        raise SystemExit(f"version {version} never passed the quality gate: no rollback to it")
    move_label(client, version, "production")


def main(argv: list[str] | None = None) -> None:
    from langfuse import Langfuse

    from legalrag.settings import get_settings

    p = argparse.ArgumentParser(description="manage the system prompt in Langfuse")
    p.add_argument("command", choices=["push", "promote", "rollback"])
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--version", type=int)
    p.add_argument("--verdict", type=Path, help="promote: the smoke gate's verdict.json")
    p.add_argument("--text-file", type=Path, help="prompt text to push (default: the code prompt)")
    p.add_argument("--prompt-version", default=PROMPT_VERSION)
    args = p.parse_args(argv)
    configure_logging("INFO")
    s = get_settings()
    client = Langfuse(public_key=s.langfuse_public_key.get_secret_value(),
                      secret_key=s.langfuse_secret_key.get_secret_value(),
                      base_url=s.langfuse_host)  # fmt: skip
    if args.command == "push":
        text = args.text_file.read_text(encoding="utf-8") if args.text_file else SYSTEM_PROMPT
        labels = check_push_labels(args.labels)
        created = client.create_prompt(name=PROMPT_NAME, prompt=text, labels=labels, type="text",
                                       config={"prompt_version": args.prompt_version},
                                       commit_message=f"prompt {args.prompt_version}")  # fmt: skip
        log.info("prompt_pushed", version=created.version, labels=labels,
                 sha256=prompt_sha256(text)[:12])  # fmt: skip
    elif args.command == "promote":
        verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
        promote(client, args.version, verdict, APPROVED)
        log.info("prompt_promoted", version=args.version)
    else:
        rollback(client, args.version, APPROVED)
        log.info("prompt_rolled_back", version=args.version)
    client.flush()


if __name__ == "__main__":
    main()
