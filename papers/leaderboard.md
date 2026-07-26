# OptimBench leaderboard (benchmark v1.0)

Task score (IQM) on 50 held-out test seeds per difficulty, higher is better. The reference (a strong deterministic solve) sits at 1.0 by definition.

| agent | easy task | medium task | hard task |
|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 |
| greedy | 0.826 | 0.746 | 0.650 |
| learned | 0.934 | 0.859 | 0.804 |

Robustness and integrity (mean over the same seeds):

| agent | easy robust / integ | medium robust / integ | hard robust / integ |
|---|---|---|---|
| random | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| greedy | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| learned | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
