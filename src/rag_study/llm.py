"""Thin multi-provider LLM client (plain HTTP) with an on-disk cache so reruns cost nothing."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import requests
import yaml


class LLM:
    def __init__(self, provider: str, model: str = "", temperature: float = 0.0,
                 cache_path: str = ".cache/llm.sqlite", responder=None, max_retries: int = 5):
        self.provider, self.model, self.temperature = provider, model, temperature
        self.cache_path, self.responder, self.max_retries = cache_path, responder, max_retries
        if cache_path != ":memory:":
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    @classmethod
    def from_config(cls, path: str, **kw):
        with open(path) as f:
            return cls(**yaml.safe_load(f), **kw)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512, tag: str = "") -> str:
        key = hashlib.sha256("\x00".join(
            [self.provider, self.model, str(self.temperature), system, prompt, str(max_tokens), tag]).encode()).hexdigest()
        row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            return row[0]
        text = self._call(prompt, system, max_tokens)
        self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, text))
        self._db.commit()
        return text

    def _post(self, url, headers, body):
        for attempt in range(12):
            r = requests.post(url, headers=headers, json=body, timeout=90)
            if r.status_code in (429, 500, 502, 503):
                wait = float(r.headers.get("retry-after", 0) or 0) or min(2 ** attempt * 2, 60)
                print(f"[{r.status_code}] waiting {wait:.0f}s: {r.text[:150]}", flush=True)
                time.sleep(min(wait + 1, 300))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"{self.provider} kept returning retryable errors (rate limit?)")

    def _call(self, prompt, system, max_tokens):
        if self.provider == "mock":
            return self.responder(prompt, system)
        if self.provider == "groq":
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            j = self._post("https://api.groq.com/openai/v1/chat/completions",
                           {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                           {"model": self.model, "messages": msgs, "temperature": self.temperature,
                                                       "max_tokens": max(max_tokens, 2048), "reasoning_effort": "low"})
            return j["choices"][0]["message"]["content"]
        if self.provider == "gemini":
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": self.temperature, "maxOutputTokens": max_tokens}}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            j = self._post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, body)
            return j["candidates"][0]["content"]["parts"][0]["text"]
        raise ValueError(f"unknown provider: {self.provider}")
