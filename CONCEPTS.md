# Concepts: prefix caching, and why the router has to know about it

`prefixrouter` is a small gateway in front of several vLLM replicas that sends requests sharing the same system prompt to the same replica, so the KV blocks for that prompt are computed once and reused. This primer covers what the cache does inside one replica, why a load balancer can silently throw the benefit away, and how to route so it doesn't.

## 1. Automatic prefix caching inside one replica

With `--enable-prefix-caching` (on by default in vLLM V1), every full KV block is tagged with a hash of its token contents *and* of every block before it in the sequence. When a new request arrives, its prompt is hashed block by block from the start; for every leading block whose hash already exists in the cache and whose contents are still resident, the KV is reused instead of recomputed. Prefill then starts at the first uncached block.

Consequences:

- **Only prefixes are shared.** Two prompts that share the first 1,500 tokens (a system prompt) and differ afterwards share those 1,500 tokens' blocks; a shared chunk in the middle shares nothing, because the hash of a block includes its predecessors.
- **The saving is prefill compute and TTFT.** A 1,500-token system prompt on a T4 with a 3B model is ~300 ms of prefill; cached, it costs the time to look up ~94 block hashes. The decode phase is unchanged.
- **Cached blocks are evictable.** They stay as long as there is free cache; under pressure they are evicted LRU-style (project 9 saw this). A busy replica keeps hot prefixes resident; a cold one has to recompute.
- **Memory is shared too.** Sixty-four concurrent requests with the same system prompt hold one copy of its blocks (kvcalc measured 5.4% cache use where naive arithmetic predicted 24%).
- **Multi-turn chat is the big win.** Turn N's prompt is turns 1..N−1 plus the new message; all of it is a prefix of itself. With the cache warm, each turn prefills only the newest message. This is where "up to 80% TTFT reduction" claims come from.

vLLM exports `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` (tokens queried and tokens found), so the hit rate per replica is observable.

## 2. The load balancer problem

Put N replicas behind a round-robin or least-connections balancer. A conversation's turns go to different replicas; each replica sees the conversation's prefix for the first time and prefills all of it. The per-replica cache is warm for *some* conversations and cold for the one in front of it, and the measured hit rate drops toward 1/N of what a single replica would achieve. The cache still works; the routing defeats it.

The same thing happens with a fleet of agents sharing one long tool-definition system prompt: with 4 replicas and random routing, each replica eventually caches the prompt, but only after each has paid the prefill once, and only while the prompt stays resident on all four, which quadruples the memory spent on it.

## 3. Prefix-aware routing

The fix is **affinity**: requests whose prompts share a long prefix should land on the same replica. Three ways to decide the key:

- **Hash of the system prompt** (this project): cheap, needs no state, works for the "many users, few system prompts" shape. Chat turns under one system prompt land together; two different system prompts may land on different replicas.
- **Session or conversation id** from a header: routes a whole conversation to one replica regardless of content; needs clients to send the id.
- **Prefix tree** of recently seen prompts per replica (SGLang's router, vLLM's production-stack router): the gateway keeps a radix tree of what each replica has cached and picks the replica with the longest match. Most precise, most state, out of scope here.

With a key in hand, mapping key → replica with **consistent hashing** keeps assignments stable when replicas come and go: each replica gets many virtual points on a ring, the key hashes to a point, the next replica clockwise owns it. Adding or removing a replica moves only ~1/N of the keys. Plain `hash(key) % N` would reshuffle everything on every change and dump every cache.

## 4. Affinity versus balance

Affinity concentrates load: a popular system prompt sends all its traffic to one replica while others idle. So the router needs a release valve: if the chosen replica is above a load threshold (queued requests, or in-flight count), fall back to the least-loaded replica and accept a cache miss. The trade-off is explicit: a miss costs one prefill (hundreds of ms); an overloaded replica costs every request on it (queueing, seconds). The threshold is where a miss is cheaper than the queue. The experiment measures both sides.

Consistent hashing with virtual nodes also spreads *different* prompts evenly, so with many system prompts the natural distribution is already balanced; the valve is for the skewed case.

## 5. What the gateway measures

The gateway exposes its own `/metrics`: requests routed by affinity vs by fallback, per-replica in-flight counts, and the replica's prefix-cache hit rate scraped from each vLLM. The experiment runs the same multi-turn chat load against two T4 replicas three ways:

1. **Round-robin** (affinity off): the baseline that most deployments have.
2. **Affinity on**: same load, routed by system-prompt hash.
3. **Affinity with the fallback valve triggering**: a skewed load where one prompt dominates.

and reports TTFT p50/p95, the prefix-cache hit rate per replica, and per-replica load, for each.

## 6. What this is not

Not a full gateway: no auth, no rate limits, no retries, no provider fallback (that is project 13, which can sit in front of this). Not a radix-tree router. Not session stickiness beyond the system prompt. Those are extensions, and the README says which would matter first.

## Further reading

- vLLM docs: *Automatic Prefix Caching* (design and the hashing rules), and the *production-stack* router's prefix-aware routing.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024), RadixAttention.
- Karger et al., *Consistent Hashing and Random Trees* (1997); DeCandia et al., *Dynamo* (2007) for virtual nodes in practice.
