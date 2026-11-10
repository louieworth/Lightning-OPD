# byte16 cluster results (Lightning OPD / OPD, 11/03-11/10)

Per-eval-step metrics copied from `outputs/<experiment>/eval_metrics/<step-1>.json` (aggregates only; no generations,
data or checkpoints). Mean = average of the four per-dataset avg@k values. Step 0 is the shared initial student.

## A: Qwen3-1.7B <- Qwen3-4B-Instruct-2507 (K=40 only here; K=1/K=120 on W&B) — aime24/25/26+hmmt26 avg@8, 32k

| step | K=40 |
|---|---|
| 0 | — |
| 20 | 0.107 |
| 40 | 0.263 |
| 60 | 0.258 |
| 80 | 0.270 |
| 100 | 0.262 |
| 120 | 0.249 |

## B: Qwen3-1.7B <- Qwen3-4B — aime24/25/26+hmmt26 avg@8, 32k

| step | K=40 |
|---|---|
| 0 | — |
| 20 | 0.111 |
| 40 | 0.132 |
| 60 | 0.098 |
| 80 | 0.114 |
| 100 | 0.120 |
| 120 | 0.117 |

## C: R1-Distill-Qwen-1.5B <- JustRL-1.5B — aime24/25/26+hmmt26 avg@8, 32k

| step | K=40 |
|---|---|
| 0 | — |
| 20 | 0.196 |
| 40 | 0.216 |
| 60 | 0.255 |
| 80 | 0.306 |
| 100 | 0.309 |
| 120 | 0.320 |

## Qwen3-0.6B-Base <- Qwen3-4B-Instruct-2507 — amc23/aime26/hmmt26/GPQA-Diamond-Physics avg@8, 16k

| step | K=1 | K=10 | K=20 | K=40 | K=120 |
|---|---|---|---|---|---|
| 0 | 0.015 | — | — | — | — |
| 20 | 0.013 | 0.019 | 0.015 | 0.016 | 0.017 |
| 40 | 0.008 | 0.011 | 0.013 | 0.017 | 0.016 |
| 60 | 0.003 | 0.017 | 0.013 | 0.012 | 0.018 |
| 80 | 0.000 | 0.009 | 0.009 | 0.015 | 0.018 |
| 100 | 0.000 | 0.073 | 0.053 | 0.080 | 0.095 |
| 120 | 0.000 | 0.077 | 0.097 | 0.111 | 0.116 |

## Qwen3-1.7B-Base <- Qwen3-4B-Instruct-2507 — amc23/aime26/hmmt26/GPQA-Diamond-Physics avg@8, 16k

| step | K=1 | K=10 | K=20 | K=40 | K=120 |
|---|---|---|---|---|---|
| 0 | 0.033 | — | — | — | — |
| 20 | 0.041 | 0.039 | 0.043 | 0.030 | 0.042 |
| 40 | 0.089 | 0.107 | 0.119 | 0.094 | 0.079 |
| 60 | 0.065 | 0.136 | 0.131 | 0.116 | 0.127 |
| 80 | 0.046 | 0.128 | 0.157 | 0.152 | 0.163 |
| 100 | 0.097 | 0.190 | 0.198 | 0.197 | 0.183 |
| 120 | 0.095 | 0.199 | 0.201 | 0.215 | 0.206 |

## Qwen3-4B-Base <- Qwen3-4B-Instruct-2507 — amc23/aime26/hmmt26/GPQA-Diamond-Physics avg@8, 16k

| step | K=1 | K=10 | K=20 | K=40 | K=120 |
|---|---|---|---|---|---|
| 0 | 0.076 | — | — | — | — |
| 20 | 0.089 | 0.113 | 0.092 | 0.088 | 0.094 |
| 40 | 0.091 | 0.033 | 0.042 | 0.046 | 0.077 |
| 60 | 0.099 | 0.189 | 0.244 | 0.245 | 0.254 |
| 80 | 0.229 | 0.358 | 0.349 | 0.358 | 0.321 |
| 100 | 0.246 | 0.372 | 0.307 | 0.349 | 0.333 |
| 120 | 0.251 | 0.374 | 0.371 | 0.342 | 0.344 |

## Initial student-teacher top-64 overlap (paper Sec. 2.2; tools/student_teacher_overlap.py, 256 rollouts)

| pair | overlap % | paper % |
|---|---|---|
| A Qwen3-1.7B <- 4B-Instruct-2507 | 62.46 | 62.5 |
| B Qwen3-1.7B <- Qwen3-4B | 70.8 | 70.0 |
| C R1-1.5B <- JustRL-1.5B | 81.14 | 79.7 |
| D R1-7B <- Light-R1-14B | 72.64 | 72.0 |
| 0.6B-Base <- 4B-Instruct-2507 | 60.27 | — |
| 1.7B-Base <- 4B-Instruct-2507 | 53.37 | — |
| 4B-Base <- 4B-Instruct-2507 | 54.2 | — |

## Findings

- A-D (post-trained students) match the paper's direction: A Semi-OPD (K=1) > OPD (last step +12.4, AUC +7.5; paper +9.5/+7.09); B tie; C mixed; D OPD better. K=40 tracks K=120.
- Base students (overlap 53-60%, lower than A) contradict the paper's overlap trend: K=1 is worst (last step -9 to -12 vs K=120; 0.6B collapses to 0). K>=10 reaches K=120 for 1.7B/4B; 0.6B needs K>=40.
- Base students' per-rollout overlap spread is 13-15% (2% for A-D): their initial rollouts are uneven in quality.
- 0.6B base K=1 (v2) has HF exports only for step 120 (earlier steps lost before the multi-node export fix).
