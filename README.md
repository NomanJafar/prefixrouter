# prefixrouter

Prefix-aware routing gateway for LLM replicas: requests sharing a system prompt land on the same replica so KV cache blocks get reused.

Project 4 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

A gateway that routes requests sharing a system prompt to the same replica so cached KV blocks are reused instead of recomputed.

## Why it matters

Reused cache is free speed: up to 80% TTFT cuts on multi-turn workloads.

## Status

Concept primer written (`CONCEPTS.md`). The proxy is next; results land in `results/` once the 2xT4 run is done.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/prefixrouter/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU experiments run from `notebooks/`; each notebook clones this repo, installs it and writes its outputs to `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
