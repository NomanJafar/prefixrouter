import asyncio
import json

import httpx

from prefixrouter.proxy import create_app


def _body(system, user="hi", stream=True):
    return {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 3, "stream": stream, "ignore_eos": True}


def test_same_system_prompt_always_lands_on_the_same_replica(two_replicas, live_proxy):
    with live_proxy(create_app(two_replicas, mode="affinity")) as url:
        seen = set()
        for i in range(6):
            r = httpx.post(f"{url}/v1/chat/completions", json=_body("SYS-A", f"turn {i}", stream=False), timeout=30)
            assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 3
            seen.add(r.headers["x-prefixrouter-replica"])
            assert r.headers["x-prefixrouter-reason"] == "affinity"
        assert len(seen) == 1


def test_round_robin_mode_alternates(two_replicas, live_proxy):
    with live_proxy(create_app(two_replicas, mode="round_robin")) as url:
        reps = [httpx.post(f"{url}/v1/chat/completions", json=_body("SYS-A", stream=False), timeout=30).headers["x-prefixrouter-replica"] for _ in range(4)]
        assert reps[0] != reps[1] and reps[0] == reps[2]


def test_streaming_is_passed_through_chunk_by_chunk(two_replicas, live_proxy):
    with live_proxy(create_app(two_replicas)) as url:
        with httpx.stream("POST", f"{url}/v1/chat/completions", json=_body("SYS-B"), timeout=30) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
        content = [json.loads(ln[5:]) for ln in lines if "[DONE]" not in ln]
        assert sum(1 for c in content if c["choices"] and c["choices"][0]["delta"].get("content")) == 3


def test_fallback_when_affinity_replica_is_over_threshold(two_replicas, live_proxy):
    app = create_app(two_replicas, mode="affinity", load_threshold=2)
    with live_proxy(app) as url:
        async def go():
            async with httpx.AsyncClient(base_url=url, timeout=30) as c:
                body = _body("SYS-C", stream=False); body["max_tokens"] = 40
                rs = await asyncio.gather(*[c.post("/v1/chat/completions", json=body) for _ in range(6)])
                return [(r.headers["x-prefixrouter-replica"], r.headers["x-prefixrouter-reason"]) for r in rs]
        decisions = asyncio.run(go())
        reasons = {d[1] for d in decisions}
        assert "affinity" in reasons and "fallback" in reasons
        assert len({d[0] for d in decisions}) == 2
        m = httpx.get(f"{url}/metrics").text
        assert "prefixrouter_decisions_total{reason=\"fallback\"}" in m and "prefixrouter_decisions_total{reason=\"affinity\"}" in m


def test_metrics_include_replica_in_flight_and_scraped_prefix_cache_rates(two_replicas, live_proxy):
    with live_proxy(create_app(two_replicas)) as url:
        httpx.post(f"{url}/v1/chat/completions", json=_body("SYS-D", stream=False), timeout=30)
        m = httpx.get(f"{url}/metrics").text
        assert "prefixrouter_replica_in_flight" in m and "prefixrouter_replica_requests_total" in m
        assert "prefixrouter_replica_prefix_cache_hit_rate" in m   # mock exports no prefix series -> NaN/absent is fine, name must exist


def test_health_and_replica_list(two_replicas, live_proxy):
    with live_proxy(create_app(two_replicas)) as url:
        h = httpx.get(f"{url}/health").json()
        assert h["status"] == "ok" and set(h["replicas"]) == set(two_replicas)
