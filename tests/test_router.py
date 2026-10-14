from prefixrouter.router import Replica, Router


def _router(threshold=4):
    return Router(["http://r1", "http://r2"], load_threshold=threshold)


def test_router_picks_affinity_replica_when_under_threshold():
    r = _router()
    key = "k"
    primary = r.ring.get(key)
    choice = r.choose(key)
    assert choice.url == primary and choice.reason == "affinity"


def test_router_falls_back_to_least_loaded_when_affinity_replica_is_busy():
    r = _router(threshold=2)
    key = "k"; primary = r.ring.get(key); other = next(u for u in r.replicas if u != primary)
    r.replicas[primary].in_flight = 2
    r.replicas[other].in_flight = 0
    choice = r.choose(key)
    assert choice.url == other and choice.reason == "fallback"


def test_router_stays_on_affinity_replica_if_every_replica_is_busy():
    r = _router(threshold=2)
    for rep in r.replicas.values(): rep.in_flight = 5
    key = "k"
    assert r.choose(key).url == r.ring.get(key)


def test_router_round_robin_mode_ignores_key():
    r = Router(["http://r1", "http://r2"], mode="round_robin")
    urls = [r.choose("same-key").url for _ in range(4)]
    assert urls == ["http://r1", "http://r2", "http://r1", "http://r2"]
    assert r.choose("same-key").reason == "round_robin"


def test_router_counts_decisions_for_metrics():
    r = _router(threshold=1)
    key = "k"; primary = r.ring.get(key)
    r.choose(key); r.replicas[primary].in_flight = 1; r.choose(key)
    assert r.counters["affinity"] == 1 and r.counters["fallback"] == 1


def test_replica_tracks_in_flight():
    rep = Replica("http://r1")
    rep.acquire(); rep.acquire(); rep.release()
    assert rep.in_flight == 1 and rep.total == 2
