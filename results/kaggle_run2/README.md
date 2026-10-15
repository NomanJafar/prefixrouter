Run 2 (2026-10-15): both replicas started, one per GPU (187,744 KV tokens each); every condition then failed with
`asyncio.run() cannot be called from a running event loop` (Jupyter owns the loop), the mistake already recorded after
quantlab run 2. Run 3 runs the session load on a worker thread.
