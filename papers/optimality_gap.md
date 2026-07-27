# Reference optimality gap

How far the heuristic reference (the task-score denominator) sits above a true OR-Tools optimum, over 20 seeds per difficulty. Smaller is tighter. The reference stays heuristic on purpose, so scoring needs no solver and is deterministic; this table is the honest disclosure of what that costs.

| difficulty | mean heuristic | mean optimal | reference above optimal |
|------------|----------------|--------------|-------------------------|
| easy | 318.3 | 279.3 | 13.8% |
| medium | 444.4 | 387.9 | 14.3% |
| hard | 625.4 | 523.9 | 19.4% |
