# OptimBench Benchmark Card

## Motivation
Language-model agents can produce answers; knowing whether they operate reliably
under constraints is unsolved. Optimization is the setting where correctness is
checkable by code: constraints are explicit, objectives are measurable, invalid
solutions are detectable, and the gap to a reference is quantifiable.

## Task definition (dynamic vehicle dispatch)
The agent operates a fleet over several turns via a tool API. It observes orders
(node, demand, time window), vehicles (capacity, in-service, route), and can
query true travel times. It assigns orders, sequences routes, and commits with
`dispatch`. Each commit triggers the next disruption wave (breakdown, which removes
the busiest in-service vehicle so the loss cannot be dodged, then rush order, then
cancellation), invalidating part of the committed plan and forcing recovery.

## Evaluation
Three deterministic scores, no model-in-the-loop judge:
- **task**: `min(1, reference_cost / committed_cost)` behind a hard feasibility gate (0 if infeasible). `reference_cost` solves the same final instance from scratch (sweep assignment, then nearest-neighbour + 2-opt routing), independent of the agent's own assignment, so the score reflects assignment *and* routing quality.
- **robustness**: fraction of committed states that were feasible, counting every disruption wave the agent was expected to face; waves left unfaced score as failures.
- **integrity**: 1.0 unless the agent spammed invalid actions, never committed, or ended with disruptions unresolved.

Feasibility (the gate) enforces: capacity, full live-order coverage, depot-anchored
routes, and shift limits on a service-time-aware schedule. Time windows are modeled
but generously bounded in v1 (see Limitations).

## Baselines
Greedy dispatcher (best-fit assignment + nearest-neighbour routing, re-planning
after each disruption), 50 scenarios per difficulty: feasibility 100% across
easy/medium/hard; task 0.83 / 0.74 / 0.65 (easy/medium/hard); robustness 1.00;
integrity 1.00. Greedy is always feasible and resolves every disruption, but its
non-geometric assignment leaves substantial task headroom that widens with
difficulty, the room a smarter agent has to improve.

## Failure modes it targets
- Premature completion: leaving a post-disruption wave unresolved is caught by
  robustness and the `disruptions_unresolved` integrity flag (which keys on feasibly
  resolved waves, not raw commit count).
- Reward gaming: dropping the depot from routes or leaving orders unassigned (blocked
  by the feasibility gate); keeping a vehicle idle to dodge the breakdown (blocked by
  targeting the busiest vehicle); routing a single well-sequenced assignment (the
  reference is an independent full solve, so a fragmented plan scores below a
  consolidated one).
- Distractor confusion: cancelled orders, out-of-service vehicles, stale traffic.

## Feasibility guarantee
Every instance is feasible by construction, and remains feasible after each
disruption: total demand is bounded so a solution exists on the remaining fleet
after a breakdown, and time windows are set from a global horizon so a
capacity-feasible plan is always time-feasible.

## Intended use and out-of-scope use
Intended: measuring whether an agent (LLM, learned policy, or heuristic) can operate a
constrained optimization problem reliably, recover from mid-episode disruptions, and reach
a low-cost feasible plan without gaming the grader. It is meant for comparing agents under
a deterministic, reproducible reward.

Out of scope: it is not a proof of optimality (the reference is a strong heuristic, not an
exact solver), not a general planning or reasoning benchmark, and not a real-world logistics
simulator (travel times, demand, and disruptions are synthetic). Scores are only comparable
within one `BENCHMARK_VERSION`.

## Instance composition
Instances are procedurally generated and seeded, so there is no collected data, no PII, and
no licensing constraint, and the pool is effectively unbounded. Each difficulty fixes a
distribution over the generation knobs:

| difficulty | nodes | vehicles | orders | capacity | slack | disruption waves | offline vehicles | stale traffic |
|------------|-------|----------|--------|----------|-------|------------------|------------------|---------------|
| easy       | 12    | 3        | 8      | 12       | 1.40  | 1                | 0                | 10%           |
| medium     | 20    | 4        | 14     | 12       | 1.25  | 2                | 1                | 15%           |
| hard       | 30    | 5        | 22     | 12       | 1.12  | 3                | 1                | 20%           |

Slack is fleet capacity over total demand (lower is tighter). As difficulty rises the fleet
gets tighter, more disruption waves hit, and more of the observed travel-time matrix is stale.

## Reproducibility
Every reported number in this repo uses the held-out test seeds `range(50)` per difficulty
(`TEST_SEEDS` in `optimbench/evaluation/splits.py`). Training draws disjoint `TRAIN_SEEDS`
and selects checkpoints on disjoint `VAL_SEEDS`, so no reported number is seen during model
selection. Reproduce the baseline table and an LLM run with:

```bash
uv pip install -e ".[dev,rl]"
python scripts/benchmark.py --seeds 50
python scripts/run_llm.py --difficulty easy --seeds 5
```

The domain rules, reference solver, and verifier are deterministic and the environment holds
no RNG, so a `(seed, difficulty, BENCHMARK_VERSION)` triple reproduces a run exactly. Pin the
numpy and torch versions from `pyproject.toml` when reporting.

## Limitations
- Time windows are loose in v1 (capacity, coverage, depot-anchoring and recovery are
  the binding constraints); binding windows that preserve the feasibility guarantee
  are planned for a later version.
- The reference is a sweep + nearest-neighbour + 2-opt heuristic, not a proven
  optimum, so `task` measures the gap to a strong baseline rather than a true
  optimality gap; an agent that beats the heuristic simply saturates at 1.0. The gap
  is quantified offline against an OR-Tools optimum (`scripts/optimality_report.py`,
  see `papers/optimality_gap.md`): the reference sits about 14% above optimal on easy
  and medium and 19% on hard, so `task` is mildly optimistic, more so on hard. The
  solver is kept out of the scoring path on purpose, to keep scoring deterministic and
  dependency-light.
- The invalid-action-spam flag is a soft signal: an agent can dilute its rejection
  rate with accepted no-ops. It cannot inflate a score (integrity also requires
  committing and feasibly resolving every wave), so it is a transparency indicator,
  not a gate.
- Single problem family (vehicle dispatch); scheduling and packing members are planned.

## Versioning and maintenance
The scoring is versioned by `BENCHMARK_VERSION` (`optimbench/domain/version.py`), currently
`1.1`. Any change to the feasibility gate, the disruption model, the reference solver, or a
score formula bumps it, which makes old and new numbers non-comparable by construction rather
than by footnote. v1.1 made the REROUTE auto-router sequence on observed (stale) travel times
rather than true ones, so it no longer bypasses the stale-traffic dimension. The planned v2
change is binding time windows (see Limitations). Issues: https://github.com/yferc/optimbench/issues.
