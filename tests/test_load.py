import asyncio

from prefixrouter.load import Session, SessionLoad, make_system_prompts, summarize_records


def test_make_system_prompts_are_distinct_and_long():
    ps = make_system_prompts(4, approx_tokens=1200)
    assert len(set(ps)) == 4 and all(1000 <= len(p.split()) <= 1400 for p in ps)


def test_session_grows_its_history_each_turn():
    s = Session(system="SYS", user_turns=["a", "b", "c"])
    m1 = s.next_messages()
    s.record_assistant("reply-1")
    m2 = s.next_messages()
    assert [m["role"] for m in m1] == ["system", "user"]
    assert [m["role"] for m in m2] == ["system", "user", "assistant", "user"] and m2[2]["content"] == "reply-1"
    assert s.turns_left == 1


def test_session_load_runs_turns_sequentially_per_session_and_tags_records(two_replicas, live_proxy):
    from prefixrouter.proxy import create_app
    with live_proxy(create_app(two_replicas, mode="affinity")) as url:
        load = SessionLoad(base_url=url, model="m", system_prompts=["S1", "S2"], sessions_per_prompt=2, turns=3,
                           max_tokens=4, user_text="hello")
        records = asyncio.run(load.run())
    assert len(records) == 2 * 2 * 3
    assert {r.meta["system_idx"] for r in records} == {0, 1}
    assert all(r.meta["replica"] in two_replicas and r.meta["reason"] == "affinity" for r in records)
    per_prompt = {i: {r.meta["replica"] for r in records if r.meta["system_idx"] == i} for i in (0, 1)}
    assert all(len(v) == 1 for v in per_prompt.values())     # each system prompt stuck to one replica


def test_summarize_records_groups_by_turn_and_replica():
    from tokenbench.metrics import RequestRecord
    recs = [RequestRecord(t_start=0, token_times=[0.1 + i * 0.01, 0.2], t_end=0.2, prompt_tokens=10, completion_tokens=2,
                          meta={"turn": t, "replica": "r1" if i % 2 else "r2", "reason": "affinity", "system_idx": 0})
            for i, t in enumerate([0, 1, 2, 0, 1, 2])]
    s = summarize_records(recs)
    assert set(s["by_turn"]) == {0, 1, 2} and s["by_turn"][0]["n"] == 2
    assert set(s["by_replica"]) == {"r1", "r2"} and s["decisions"] == {"affinity": 6}
    assert s["ttft"]["n"] == 6
