"""Prompts by label: the served system prompt comes from Langfuse, falls back to the code."""

from __future__ import annotations

from types import SimpleNamespace

from legalrag.generation import PROMPT_VERSION, SYSTEM_PROMPT, Generator, build_messages
from legalrag.observability.prompts import PROMPT_NAME, CodePrompt, LangfusePrompt, move_label
from legalrag.pipeline import RagPipeline
from tests.fakes import FakeLLM
from tests.unit.test_generation_pipeline import StubRetriever, chunk


class FakePrompts:
    def __init__(self, text="Rules v4", version=2, config=None, fail=False):
        self.text, self.version, self.config, self.fail = text, version, config, fail
        self.calls, self.updates = [], []

    def get_prompt(self, name, **kw):
        self.calls.append((name, kw))
        if self.fail:
            raise ConnectionError("langfuse down")
        return SimpleNamespace(
            prompt=self.text,
            version=self.version,
            is_fallback=False,
            config=self.config or {"prompt_version": "v4"},
        )

    def update_prompt(self, **kw):
        self.updates.append(kw)


def test_the_label_decides_which_prompt_is_served():
    client = FakePrompts()
    served = LangfusePrompt(client, label="production").get()
    assert (served.text, served.version, served.source) == ("Rules v4", "v4", "langfuse:2")
    name, kw = client.calls[0]
    assert name == PROMPT_NAME and kw["label"] == "production" and kw["cache_ttl_seconds"] == 60


def test_an_unreachable_langfuse_serves_the_prompt_in_the_code():
    served = LangfusePrompt(FakePrompts(fail=True), label="production").get()
    assert (served.text, served.version, served.source) == (SYSTEM_PROMPT, PROMPT_VERSION, "code")


def test_the_pipeline_sends_the_served_prompt_and_reports_its_version():
    llm = FakeLLM("Yes [Art. 147].")
    pipe = RagPipeline(
        StubRetriever([chunk(147)]),
        Generator(llm, "m", 100, 0.0),
        prompts=LangfusePrompt(FakePrompts(), label="production"),
    )
    answer = asyncio_run(pipe.ask("Is a contract binding?"))
    assert llm.calls[0]["messages"][0]["content"] == "Rules v4"
    assert answer.prompt_version == "v4"


def test_code_prompt_is_the_default():
    served = CodePrompt().get()
    assert served.text == SYSTEM_PROMPT and served.version == PROMPT_VERSION
    assert build_messages("q?", [chunk(1)], system_prompt="X")[0]["content"] == "X"


def test_promotion_and_rollback_are_a_label_move():
    client = FakePrompts()
    move_label(client, version=1, label="production")
    assert client.updates == [{"name": PROMPT_NAME, "version": 1, "new_labels": ["production"]}]


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


class VersionedPrompts(FakePrompts):
    """get_prompt(version=N) returns that version's text, like Langfuse."""

    def __init__(self, texts):
        super().__init__()
        self.texts = texts

    def get_prompt(self, name, **kw):
        n = kw["version"]
        return SimpleNamespace(
            prompt=self.texts[n],
            version=n,
            is_fallback=False,
            config={"prompt_version": f"v{n + 2}"},
        )


def verdict_for(text, passed=True):
    from legalrag.observability.prompts import prompt_sha256

    return {"passed": passed, "prompt_sha256": prompt_sha256(text), "faithfulness": 0.95}


def test_promotion_needs_a_passing_verdict_for_exactly_that_text(tmp_path):
    import pytest

    from legalrag.observability.prompts import promote

    client = VersionedPrompts({1: "rules v3", 2: "rules v4"})
    approved = tmp_path / "approved.json"
    with pytest.raises(SystemExit, match="did not pass"):
        promote(client, 2, verdict_for("rules v4", passed=False), approved)
    with pytest.raises(SystemExit, match="different prompt"):
        promote(client, 2, verdict_for("rules v3"), approved)  # the verdict is for another text
    assert client.updates == []
    promote(client, 2, verdict_for("rules v4"), approved)
    assert client.updates == [{"name": PROMPT_NAME, "version": 2, "new_labels": ["production"]}]


def test_rollback_only_to_a_version_that_passed_before(tmp_path):
    import pytest

    from legalrag.observability.prompts import promote, rollback

    client = VersionedPrompts({1: "rules v3", 2: "rules v4", 3: "never gated"})
    approved = tmp_path / "approved.json"
    promote(client, 1, verdict_for("rules v3"), approved)
    promote(client, 2, verdict_for("rules v4"), approved)
    rollback(client, 1, approved)  # back to a gated version: allowed, no new evaluation needed
    assert client.updates[-1]["version"] == 1
    with pytest.raises(SystemExit, match="never passed"):
        rollback(client, 3, approved)


def test_new_versions_are_pushed_as_candidates_only():
    import pytest

    from legalrag.observability.prompts import check_push_labels

    assert check_push_labels([]) == ["candidate"]
    with pytest.raises(SystemExit, match="production"):
        check_push_labels(["production"])
