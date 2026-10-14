from collections import Counter

from prefixrouter.ring import ConsistentHashRing, affinity_key


def test_affinity_key_is_hash_of_system_prompt_only():
    a = affinity_key([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hi"}])
    b = affinity_key([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "something else"}])
    c = affinity_key([{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "hi"}])
    assert a == b and a != c and len(a) == 16


def test_affinity_key_without_system_prompt_uses_first_user_message_prefix():
    k1 = affinity_key([{"role": "user", "content": "Summarise: " + "x" * 500 + " tail A"}])
    k2 = affinity_key([{"role": "user", "content": "Summarise: " + "x" * 500 + " tail B"}])
    assert k1 == k2   # first 256 chars decide


def test_ring_is_deterministic_and_balanced():
    ring = ConsistentHashRing(["r1", "r2", "r3"], replicas_per_node=100)
    keys = [f"key-{i}" for i in range(3000)]
    owners = Counter(ring.get(k) for k in keys)
    assert set(owners) == {"r1", "r2", "r3"}
    assert all(800 <= n <= 1200 for n in owners.values())
    assert [ring.get(k) for k in keys[:20]] == [ring.get(k) for k in keys[:20]]


def test_removing_a_node_moves_only_its_keys():
    keys = [f"key-{i}" for i in range(3000)]
    before = {k: ConsistentHashRing(["r1", "r2", "r3"]).get(k) for k in keys}
    after = {k: ConsistentHashRing(["r1", "r2"]).get(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k] and before[k] != "r3")
    assert moved == 0
    assert all(after[k] in ("r1", "r2") for k in keys)


def test_ring_preference_order_lists_every_node_once():
    ring = ConsistentHashRing(["a", "b", "c"])
    order = ring.preference("some-key")
    assert sorted(order) == ["a", "b", "c"] and order[0] == ring.get("some-key")
