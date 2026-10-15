Run 1 (2026-10-15): every condition failed with the second replica unable to start: `Free memory on device cuda:0
(1.73/14.56 GiB) ... less than desired GPU memory utilization`. Both replicas were spawned on the same GPU: the notebook set
CUDA_VISIBLE_DEVICES while *constructing* the servers but spawned them afterwards, so both inherited the last value.
Fix for run 2: servellm's VllmServer takes a per-process `env`, and the notebook passes CUDA_VISIBLE_DEVICES per replica.
The results cell also lacked a guard for an empty table.
