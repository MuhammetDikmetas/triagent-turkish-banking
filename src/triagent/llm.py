"""Thin LLM client over any OpenAI-compatible endpoint (Gemini, Groq, Ollama, OpenAI).

Guarantees:
* Only masked text is ever sent (the caller masks; we double-check here).
* Output is parsed with Pydantic; malformed output -> one retry -> ``None`` (caller falls back).
* Responses are cached on disk by content hash, so re-running the evaluation is free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from pydantic import ValidationError

from triagent.config import ROOT, Settings, get_settings
from triagent.pii import detect
from triagent.prompts import DRAFT_SYSTEM, TRIAGE_SYSTEM, draft_user_prompt
from triagent.schemas import LLMTriage

log = logging.getLogger(__name__)
CACHE_DIR = ROOT / "eval" / "cache"


class PIILeakError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None, use_cache: bool = True):
        self.s = settings or get_settings()
        self.use_cache = use_cache
        self._client = None
        self.calls = 0  # real network calls, for cost reporting

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def _get(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.s.llm_api_key or "ollama",
                base_url=self.s.llm_base_url,
                timeout=30,
                max_retries=2,
            )
        return self._client

    # ------------------------------------------------------------------ plumbing
    def _cache_path(self, kind: str, payload: str) -> Path:
        key = hashlib.sha256(f"{self.s.llm_model}|{kind}|{payload}".encode()).hexdigest()[:24]
        return CACHE_DIR / f"{kind}_{key}.json"

    def _chat(self, system: str, user: str, json_mode: bool, kind: str) -> str:
        if detect(user):
            raise PIILeakError("Refusing to send unmasked PII to the LLM")
        path = self._cache_path(kind, user)
        if self.use_cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["content"]

        # Anthropic's OpenAI-compatible endpoint rejects {"type": "json_object"}; there we rely
        # on the prompt + Pydantic validation (+ one retry) instead.
        use_json_mode = json_mode and self.s.llm_provider != "anthropic"
        kwargs = {"response_format": {"type": "json_object"}} if use_json_mode else {}
        t0 = time.perf_counter()
        resp = self._get().chat.completions.create(
            model=self.s.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            **kwargs,
        )
        self.calls += 1
        content = resp.choices[0].message.content or ""
        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"content": content, "latency_s": round(time.perf_counter() - t0, 3)},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return content

    # ------------------------------------------------------------------ public API
    def triage(self, masked_text: str) -> LLMTriage | None:
        user = f'Müşteri mesajı:\n"""{masked_text}"""'
        for attempt in range(2):
            try:
                raw = self._chat(TRIAGE_SYSTEM, user, json_mode=True, kind=f"triage{attempt}")
                match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate ```json fences / extra text
                return LLMTriage.model_validate(json.loads(match.group() if match else raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                log.warning("LLM returned invalid JSON (attempt %d): %s", attempt + 1, exc)
            except PIILeakError:
                raise
            except Exception as exc:  # network, auth, rate limit...
                log.warning("LLM call failed: %s", exc)
                return None
        return None

    def draft(self, masked_text: str, category: str, priority: str) -> str | None:
        try:
            return self._chat(
                DRAFT_SYSTEM,
                draft_user_prompt(masked_text, category, priority),
                json_mode=False,
                kind="draft",
            ).strip()
        except PIILeakError:
            raise
        except Exception as exc:
            log.warning("LLM draft failed: %s", exc)
            return None
