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
        return SimpleNamespace(prompt=self.text, version=self.version, is_fallback=False,
                               config=self.config or {"prompt_version": "v4"})  # fmt: skip

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
    pipe = RagPipeline(StubRetriever([chunk(147)]), Generator(llm, "m", 100, 0.0),
                       prompts=LangfusePrompt(FakePrompts(), label="production"))  # fmt: skip
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
