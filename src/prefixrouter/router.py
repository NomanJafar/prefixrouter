"""Routing decision: affinity replica unless it is above the load threshold, then the
least-loaded replica (CONCEPTS.md section 4). Round-robin mode is the baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from prefixrouter.ring import ConsistentHashRing


@dataclass
class Replica:
    url: str
    in_flight: int = 0
    total: int = 0
    errors: int = 0

    def acquire(self) -> None:
        self.in_flight += 1
        self.total += 1

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


@dataclass(frozen=True)
class Choice:
    url: str
    reason: str  # affinity | fallback | round_robin


@dataclass
class Router:
    urls: list[str]
    mode: str = "affinity"            # affinity | round_robin
    load_threshold: int = 8           # in-flight requests above which affinity yields
    replicas: dict[str, Replica] = field(init=False)
    ring: ConsistentHashRing = field(init=False)
    counters: Counter = field(default_factory=Counter)
    _rr: int = 0

    def __post_init__(self):
        self.replicas = {u: Replica(u) for u in self.urls}
        self.ring = ConsistentHashRing(self.urls)

    def choose(self, key: str) -> Choice:
        if self.mode == "round_robin":
            url = self.urls[self._rr % len(self.urls)]
            self._rr += 1
            self.counters["round_robin"] += 1
            return Choice(url, "round_robin")
        primary = self.ring.get(key)
        if self.replicas[primary].in_flight < self.load_threshold:
            self.counters["affinity"] += 1
            return Choice(primary, "affinity")
        least = min(self.replicas.values(), key=lambda r: r.in_flight)
        if least.in_flight >= self.load_threshold:
            self.counters["affinity"] += 1      # everyone is busy; a miss would not help
            return Choice(primary, "affinity")
        self.counters["fallback"] += 1
        return Choice(least.url, "fallback")
