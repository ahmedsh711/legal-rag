"""Test doubles shared by unit and integration tests."""

from __future__ import annotations

from types import SimpleNamespace


class FakeStream:
    """Mimics openai.AsyncStream: async-iterable chunks plus close()."""

    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    async def __aiter__(self):
        for c in self.chunks:
            yield c

    async def close(self):
        self.closed = True


class FakeLLM:
    """Mimics openai.AsyncOpenAI: chat.completions.create(...) returns a reply or a stream."""

    def __init__(self, reply: str, no_choices: bool = False):
        self.reply, self.no_choices = reply, no_choices
        self.calls, self.streams = [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_sync))

    def _create_sync(self, **kwargs):
        self.calls.append(kwargs)  # recorded at call time, even if the coroutine is never awaited
        return self._create(**kwargs)

    async def _create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
        if kwargs.get("stream"):
            chunks = [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=word + " "))],
                    usage=None,
                    model="fake-model",
                )
                for word in self.reply.split(" ")
            ]
            chunks.append(SimpleNamespace(choices=[], usage=usage, model="fake-model"))
            stream = FakeStream(chunks)
            self.streams.append(stream)
            return stream
        msg = SimpleNamespace(content=self.reply)
        choices = [] if self.no_choices else [SimpleNamespace(message=msg)]
        return SimpleNamespace(choices=choices, usage=usage, model="fake-model")
