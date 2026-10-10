"""Prompts by label: which system prompt is live is a pointer in Langfuse, not a code deploy.

    uv run python -m legalrag.observability.prompts push --labels production   # code prompt -> v1
    uv run python -m legalrag.observability.prompts move --version 2 --label production  # promote
    uv run python -m legalrag.observability.prompts move --version 1 --label production  # roll back

Langfuse keeps every version of ``legal-rag-system``; a *label* (``production``, ``canary``)
points at one of them. The API asks for the version behind its label (``PROMPT_LABEL``) through
the SDK's cache (60 s), so moving the label changes what is served within a minute, with no
build and no restart; moving it back is the rollback. If Langfuse cannot be reached, the prompt
in the code is served (``generation.SYSTEM_PROMPT``), so Langfuse is never a single point of
failure for answers.

The rule that keeps this safe: a prompt version only gets ``production`` after the quality gate
passed with it (``eval.smoke`` / the golden run). Each version carries its ``prompt_version`` in
its config, which is what /metadata, metrics, events and traces report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from legalrag.generation import PROMPT_VERSION, SYSTEM_PROMPT
from legalrag.logging_conf import configure_logging, get_logger

log = get_logger(__name__)

PROMPT_NAME = "legal-rag-system"


@dataclass(frozen=True)
class ServedPrompt:
    text: str
    version: str  # e.g. "v3": the evaluated prompt version, not Langfuse's counter
    source: str  # "code" or "langfuse:<n>"
    client: Any = None  # the Langfuse prompt object, so traces link generations to it


class CodePrompt:
    def get(self) -> ServedPrompt:
        return ServedPrompt(SYSTEM_PROMPT, PROMPT_VERSION, "code")


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


def move_label(client: Any, version: int, label: str) -> None:
    """Point ``label`` at ``version`` (Langfuse removes it from the version that had it)."""
    client.update_prompt(name=PROMPT_NAME, version=version, new_labels=[label])


def main(argv: list[str] | None = None) -> None:
    from langfuse import Langfuse

    from legalrag.settings import get_settings

    p = argparse.ArgumentParser(description="manage the system prompt in Langfuse")
    p.add_argument("command", choices=["push", "move"])
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--version", type=int)
    p.add_argument("--label", default="production")
    p.add_argument("--text-file", help="prompt text to push (default: the code prompt)")
    p.add_argument("--prompt-version", default=PROMPT_VERSION)
    args = p.parse_args(argv)
    configure_logging("INFO")
    s = get_settings()
    client = Langfuse(public_key=s.langfuse_public_key.get_secret_value(),
                      secret_key=s.langfuse_secret_key.get_secret_value(),
                      base_url=s.langfuse_host)  # fmt: skip
    if args.command == "push":
        text = open(args.text_file, encoding="utf-8").read() if args.text_file else SYSTEM_PROMPT
        created = client.create_prompt(name=PROMPT_NAME, prompt=text, labels=args.labels,
                                       type="text", config={"prompt_version": args.prompt_version},
                                       commit_message=f"prompt {args.prompt_version}")  # fmt: skip
        log.info("prompt_pushed", version=created.version, labels=args.labels)
    else:
        move_label(client, args.version, args.label)
        log.info("prompt_label_moved", version=args.version, label=args.label)
    client.flush()


if __name__ == "__main__":
    main()
