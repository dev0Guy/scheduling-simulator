# Tournament variants vs SJF, 1 machine: win % without ties

| method | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Old 10-job model, tournament eval (no extra training) | - | 60.6 | 60.5 | 61.7 | 61.4 | 60.2 | 59.5 | 61.1 | - | - |
| Trained on 1-10 jobs, tournament eval | 63.9 | 55.6 | 54.3 | 55.3 | 54.9 | 53.8 | 52.1 | 54.7 | 52.2 | 47.8 |
| End-to-end from scratch, v1 (collapsed) | - | 39.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| End-to-end from scratch, v2 (fixed) | - | 54.7 | 33.6 | 12.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Fine-tuned from 10-job model | - | 46.5 | 55.0 | - | 48.6 | - | - | - | - | - |
| Fine-tuned, best checkpoint | - | 43.3 | 53.9 | - | 48.3 | - | - | - | - | - |
| Fine-tuned, round 2 | - | 36.5 | 39.9 | - | 30.0 | - | - | - | - | - |
| Fine-tuned + critic warm-up | - | 62.1 | 60.4 | - | 60.2 | 58.0 | 55.7 | 60.0 | 52.2 | 55.3 |
