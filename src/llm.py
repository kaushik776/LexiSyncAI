"""Thin LLM client: local Ollama by default, or any OpenAI-compatible API (e.g. Groq)
for cloud deployment. Every call returns parsed JSON that follows a given schema, and
records tokens and seconds so we can report latency and cost per question."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from src import config


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    log: list[dict] = field(default_factory=list)   # one entry per call: purpose, tokens, seconds

    def add(self, purpose: str, prompt: int, completion: int, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.seconds += seconds
        self.log.append({"purpose": purpose, "prompt_tokens": prompt,
                         "completion_tokens": completion, "seconds": round(seconds, 2)})


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)        # tolerate text around the JSON
        if m:
            return json.loads(m.group(0))
        raise LLMError(f"LLM did not return JSON: {text[:200]}")


class LLM:
    def __init__(self, backend: str = config.LLM_BACKEND):
        self.backend = backend
        self.model = config.LLM_MODEL if backend == "ollama" else config.OPENAI_MODEL

    def chat_json(self, system: str, user: str, schema: dict, usage: Usage | None = None,
                  purpose: str = "answer", max_tokens: int = 1024) -> dict:
        t = time.time()
        if self.backend == "ollama":
            text, p_tok, c_tok = self._ollama(system, user, schema, max_tokens)
        else:
            text, p_tok, c_tok = self._openai(system, user, schema, max_tokens)
        if usage is not None:
            usage.add(purpose, p_tok, c_tok, time.time() - t)
        return _parse_json(text)

    def _ollama(self, system, user, schema, max_tokens):
        import ollama
        try:
            client = ollama.Client(host=config.OLLAMA_HOST)
            r = client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema,                                   # structured output (JSON schema)
                options={"temperature": config.LLM_TEMPERATURE, "seed": config.SEED,
                         "num_ctx": config.LLM_NUM_CTX, "num_predict": max_tokens},
                keep_alive="30m",
            )
        except Exception as e:                                   # connection refused, model missing ...
            raise LLMError(f"Ollama call failed ({e}). Is Ollama running and is "
                           f"'{self.model}' pulled?") from e
        return r.message.content, r.prompt_eval_count or 0, r.eval_count or 0

    def _openai(self, system, user, schema, max_tokens):
        import requests
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("Set OPENAI_API_KEY (or GROQ_API_KEY) to use the API backend.")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system + "\nReply with JSON matching this schema: "
                          + json.dumps(schema)}, {"role": "user", "content": user}],
            "temperature": config.LLM_TEMPERATURE, "seed": config.SEED, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(f"{config.OPENAI_BASE_URL}/chat/completions", json=body, timeout=120,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            raise LLMError(f"API error {r.status_code}: {r.text[:300]}")
        data = r.json()
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
