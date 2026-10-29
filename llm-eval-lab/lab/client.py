"""Thin wrapper around Ollama's chat API that returns the output plus timing metrics."""
from dataclasses import dataclass, asdict
from functools import lru_cache

import ollama

NS = 1e9  # Ollama reports durations in nanoseconds


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    output_tokens: int
    total_s: float
    load_s: float
    prompt_eval_s: float
    eval_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=None)
def _client(host: str | None) -> ollama.Client:
    return ollama.Client(host=host)


def chat(model: str, messages: list[dict], options: dict | None = None,
         host: str | None = None) -> ChatResult:
    resp = _client(host).chat(model=model, messages=messages, options=options or {})
    return ChatResult(
        content=resp.message.content or "",
        prompt_tokens=resp.prompt_eval_count or 0,
        output_tokens=resp.eval_count or 0,
        total_s=(resp.total_duration or 0) / NS,
        load_s=(resp.load_duration or 0) / NS,
        prompt_eval_s=(resp.prompt_eval_duration or 0) / NS,
        eval_s=(resp.eval_duration or 0) / NS,
    )
