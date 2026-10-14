"""Multi-turn session load for measuring prefix affinity (CONCEPTS.md section 5).

K distinct long system prompts, S sessions per prompt, T turns per session. Turns
within a session run sequentially (each turn's prompt is the whole history so far);
sessions run concurrently. Every record is tagged with the replica and the routing
reason from the gateway's response headers."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import httpx
from tokenbench.metrics import RequestRecord, latency_summary

_WORDS = ["policy", "guideline", "procedure", "customer", "account", "invoice", "ticket", "escalation", "refund", "warranty", "shipment", "region", "catalogue", "tier", "priority", "contract", "renewal", "compliance", "audit", "template", "clause"]


def make_system_prompts(n: int, approx_tokens: int = 1200, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        words = [f"SYSTEM-PROMPT-{i}:"] + [rng.choice(_WORDS) for _ in range(approx_tokens - 1)]
        out.append(" ".join(words))
    return out


@dataclass
class Session:
    system: str
    user_turns: list[str]
    history: list[dict] = field(default_factory=list)

    @property
    def turns_left(self) -> int:
        return len(self.user_turns) - sum(1 for m in self.history if m["role"] == "user")

    def next_messages(self) -> list[dict]:
        turn = sum(1 for m in self.history if m["role"] == "user")
        self.history.append({"role": "user", "content": self.user_turns[turn]})
        return [{"role": "system", "content": self.system}, *self.history]

    def record_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text or "(no output)"})


@dataclass
class SessionLoad:
    base_url: str
    model: str
    system_prompts: list[str]
    sessions_per_prompt: int = 4
    turns: int = 4
    max_tokens: int = 48
    user_text: str = "Continue the conversation with one more short paragraph."
    extra_headers: dict | None = None

    async def _run_session(self, client: httpx.AsyncClient, system_idx: int, session_idx: int, out: list[RequestRecord]) -> None:
        s = Session(self.system_prompts[system_idx], [f"{self.user_text} (turn {t + 1})" for t in range(self.turns)])
        for turn in range(self.turns):
            messages = s.next_messages()
            rec = await _stream_messages(client, self.model, messages, self.max_tokens, self.extra_headers)
            rec.meta.update(system_idx=system_idx, session_idx=session_idx, turn=turn,
                            replica=rec.meta.get("headers", {}).get("x-prefixrouter-replica"),
                            reason=rec.meta.get("headers", {}).get("x-prefixrouter-reason"))
            out.append(rec)
            s.record_assistant(rec.meta.get("text", ""))

    async def run(self) -> list[RequestRecord]:
        out: list[RequestRecord] = []
        async with httpx.AsyncClient(base_url=self.base_url, timeout=None) as client:
            await asyncio.gather(*[self._run_session(client, i, j, out)
                                   for i in range(len(self.system_prompts)) for j in range(self.sessions_per_prompt)])
        return out


async def _stream_messages(client, model, messages, max_tokens, extra_headers) -> RequestRecord:
    """Like tokenbench.client.stream_request but with an explicit message list; keeps generated text."""
    import time

    from tokenbench.client import parse_sse_line

    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    token_times, text, usage, status, headers, error = [], [], None, None, {}, None
    t0 = time.monotonic()
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=extra_headers, timeout=None) as resp:
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            if status != 200:
                error = f"HTTP {status}"
            else:
                async for line in resp.aiter_lines():
                    obj = parse_sse_line(line)
                    if obj is None:
                        continue
                    for ch in obj.get("choices", []):
                        d = (ch.get("delta") or {}).get("content")
                        if d:
                            token_times.append(time.monotonic()); text.append(d)
                    if obj.get("usage"):
                        usage = obj["usage"]
    except httpx.HTTPError as e:
        error = f"{type(e).__name__}: {e}"
    return RequestRecord(t_start=t0, token_times=token_times, t_end=time.monotonic(),
                         prompt_tokens=usage.get("prompt_tokens") if usage else None,
                         completion_tokens=usage.get("completion_tokens") if usage else len(token_times),
                         error=error, meta={"status": status, "headers": headers, "text": "".join(text)})


def summarize_records(records: list[RequestRecord]) -> dict:
    ok = [r for r in records if r.ok]
    by_turn: dict[int, dict] = {}
    for t in sorted({r.meta.get("turn") for r in ok}):
        rs = [r for r in ok if r.meta.get("turn") == t]
        by_turn[t] = {"n": len(rs), "ttft": latency_summary([r.ttft_s for r in rs]), "prompt_tokens_mean": sum(r.prompt_tokens or 0 for r in rs) / len(rs)}
    by_replica: dict[str, dict] = {}
    for rep in sorted({r.meta.get("replica") for r in ok}, key=str):
        rs = [r for r in ok if r.meta.get("replica") == rep]
        by_replica[rep] = {"n": len(rs), "ttft": latency_summary([r.ttft_s for r in rs])}
    decisions: dict[str, int] = {}
    for r in records:
        decisions[r.meta.get("reason")] = decisions.get(r.meta.get("reason"), 0) + 1
    return {"requests": len(records), "errors": len(records) - len(ok), "ttft": latency_summary([r.ttft_s for r in ok]),
            "itl": latency_summary([i for r in ok for i in r.itls_s]), "by_turn": by_turn, "by_replica": by_replica, "decisions": decisions}
