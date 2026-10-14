"""The gateway: an OpenAI-compatible proxy that routes by system-prompt affinity.

Streams are passed through chunk by chunk. Each response carries two headers,
x-prefixrouter-replica and x-prefixrouter-reason, so load tests can attribute every
request. /metrics exposes the router's decisions, per-replica load, and each
replica's prefix-cache hit rate scraped from its own /metrics.
"""

from __future__ import annotations

import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from prefixrouter.ring import affinity_key
from prefixrouter.router import Router

_PROM = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)


def _scrape(text: str, name: str) -> float | None:
    vals = [float(v) for n, v in _PROM.findall(text) if n == name]
    return sum(vals) if vals else None


def create_app(replica_urls: list[str], mode: str = "affinity", load_threshold: int = 8,
               scrape_interval_s: float = 2.0) -> FastAPI:
    router = Router(list(replica_urls), mode=mode, load_threshold=load_threshold)
    app = FastAPI(title="prefixrouter")
    client = httpx.AsyncClient(timeout=None)
    cache_rates: dict[str, dict] = {u: {"hit_rate": None, "queries": None, "hits": None, "t": 0.0} for u in replica_urls}

    async def refresh_cache_rates() -> None:
        now = time.monotonic()
        for u, rec in cache_rates.items():
            if now - rec["t"] < scrape_interval_s:
                continue
            rec["t"] = now
            try:
                r = await client.get(f"{u}/metrics", timeout=3.0)
                q, h = _scrape(r.text, "vllm:prefix_cache_queries_total"), _scrape(r.text, "vllm:prefix_cache_hits_total")
                rec.update(queries=q, hits=h, hit_rate=(h / q if q else None))
            except httpx.HTTPError:
                pass

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": router.mode, "replicas": list(router.replicas)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        await refresh_cache_rates()
        lines = [f'prefixrouter_decisions_total{{reason="{k}"}} {v}' for k, v in router.counters.items()]
        for u, rep in router.replicas.items():
            lines += [f'prefixrouter_replica_in_flight{{replica="{u}"}} {rep.in_flight}',
                      f'prefixrouter_replica_requests_total{{replica="{u}"}} {rep.total}',
                      f'prefixrouter_replica_errors_total{{replica="{u}"}} {rep.errors}']
            hr = cache_rates[u]["hit_rate"]
            lines.append(f'prefixrouter_replica_prefix_cache_hit_rate{{replica="{u}"}} {hr if hr is not None else "NaN"}')
        return "\n".join(lines) + "\n"

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        choice = router.choose(affinity_key(body.get("messages", [])))
        rep = router.replicas[choice.url]
        rep.acquire()
        headers = {"x-prefixrouter-replica": choice.url, "x-prefixrouter-reason": choice.reason}
        upstream = client.build_request("POST", f"{choice.url}/v1/chat/completions", json=body)
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            rep.errors += 1
            rep.release()
            return JSONResponse({"error": f"upstream unreachable: {type(e).__name__}"}, status_code=502, headers=headers)
        if resp.status_code != 200:
            rep.errors += 1
            data = await resp.aread()
            await resp.aclose()
            rep.release()
            return JSONResponse({"error": data.decode(errors="replace")[:500]}, status_code=resp.status_code, headers=headers)
        if body.get("stream"):
            async def gen():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    rep.release()
            return StreamingResponse(gen(), media_type=resp.headers.get("content-type", "text/event-stream"), headers=headers)
        data = await resp.aread()
        await resp.aclose()
        rep.release()
        return JSONResponse(httpx.Response(200, content=data).json(), headers=headers)

    @app.get("/v1/models")
    async def models():
        u = next(iter(router.replicas))
        r = await client.get(f"{u}/v1/models", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)

    app.state.router = router
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="prefix-affinity router for OpenAI-compatible replicas")
    p.add_argument("--replica", action="append", required=True, help="replica base URL (repeat)")
    p.add_argument("--mode", choices=["affinity", "round_robin"], default="affinity")
    p.add_argument("--load-threshold", type=int, default=8)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    uvicorn.run(create_app(a.replica, a.mode, a.load_threshold), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
