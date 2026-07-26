# OptimBench leaderboard (benchmark v1.1)

Task score (IQM) on 50 held-out test seeds per difficulty, higher is better. The reference (a strong deterministic solve) sits at 1.0 by definition.

| agent | easy task | medium task | hard task |
|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 |
| greedy | 0.825 | 0.739 | 0.648 |
| learned | 0.933 | 0.856 | 0.792 |

Robustness and integrity (mean over the same seeds):

| agent | easy robust / integ | medium robust / integ | hard robust / integ |
|---|---|---|---|
| random | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| greedy | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| learned | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
