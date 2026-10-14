"""Affinity keys and a consistent-hash ring (CONCEPTS.md section 3)."""

from __future__ import annotations

import bisect
import hashlib


def _h(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def affinity_key(messages: list[dict], prefix_chars: int = 256) -> str:
    """Hash of the system prompt; without one, of the first user message's leading chars."""
    system = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    if system:
        basis = "\n".join(str(c) for c in system)
    else:
        first = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
        basis = first[:prefix_chars]
    return hashlib.blake2b(basis.encode(), digest_size=8).hexdigest()


class ConsistentHashRing:
    def __init__(self, nodes: list[str], replicas_per_node: int = 100):
        self.replicas_per_node = replicas_per_node
        self._ring: list[tuple[int, str]] = []
        for n in nodes:
            for i in range(replicas_per_node):
                self._ring.append((_h(f"{n}#{i}"), n))
        self._ring.sort()
        self._points = [p for p, _ in self._ring]

    @property
    def nodes(self) -> list[str]:
        return sorted({n for _, n in self._ring})

    def get(self, key: str) -> str:
        if not self._ring:
            raise ValueError("empty ring")
        i = bisect.bisect(self._points, _h(key)) % len(self._ring)
        return self._ring[i][1]

    def preference(self, key: str) -> list[str]:
        """Nodes in ring order starting at the key's owner, each once: the fallback order."""
        if not self._ring:
            return []
        start = bisect.bisect(self._points, _h(key)) % len(self._ring)
        seen: list[str] = []
        for j in range(len(self._ring)):
            n = self._ring[(start + j) % len(self._ring)][1]
            if n not in seen:
                seen.append(n)
        return seen
