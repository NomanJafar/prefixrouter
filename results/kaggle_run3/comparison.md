| condition | requests | errors | ttft_p50_ms | ttft_p95_ms | turn0_ttft_p50_ms | turn1_ttft_p50_ms | turn2_ttft_p50_ms | turn3_ttft_p50_ms | hit_rate_r1 | hit_rate_r2 | split | decisions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | 64 | 0 | 203 | 2198 | 2139 | 181 | 175 | 187 | 0.87 | 0.87 | 32/32 | {"round_robin": 64} |
| affinity | 64 | 0 | 189 | 1318 | 1168 | 182 | 164 | 187 | 0.94 | 0.93 | 32/32 | {"affinity": 48, "fallback": 16} |
| affinity, skewed (one prompt x3) | 64 | 0 | 199 | 1317 | 1310 | 172 | 174 | 202 | 0.92 | 0.95 | 24/40 | {"fallback": 24, "affinity": 40} |
