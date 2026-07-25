# OptimBench — Benchmark Card

## Motivation
Language-model agents can produce answers; knowing whether they operate reliably
under constraints is unsolved. Optimization is the setting where correctness is
checkable by code: constraints are explicit, objectives are measurable, invalid
solutions are detectable, and the gap to a reference is quantifiable.

## Task definition (dynamic vehicle dispatch)
The agent operates a fleet over several turns via a tool API. It observes orders
(node, demand, time window), vehicles (capacity, in-service, route), and can
query true travel times. It assigns orders, sequences routes, and commits with
`dispatch`. Each commit triggers the next disruption wave (breakdown, rush order,
cancellation), invalidating part of the committed plan and forcing recovery.

## Evaluation
Three deterministic scores, no model-in-the-loop judge:
- **task** — `min(1, reference_cost / committed_cost)` behind a hard feasibility gate (0 if infeasible).
- **robustness** — fraction of post-disruption committed states that were feasible.
- **integrity** — 1.0 unless the agent spammed invalid actions or never committed.

Feasibility (the gate) enforces: capacity, full live-order coverage, depot-anchored
routes, delivery time windows on a service-time-aware schedule, and shift limits.

## Baselines
Greedy dispatcher (best-fit assignment + nearest-neighbour routing, re-planning
after each disruption), 50 scenarios per difficulty: feasibility 100% across
easy/medium/hard; task 0.99–1.00; robustness 1.00; integrity 1.00. The floor is
high by design — the task is solvable; the signal is where agents fall below it.

## Failure modes it targets
- Premature completion: declaring done before repairing a post-disruption violation.
- Reward gaming: dropping the depot from routes, or leaving orders unassigned, to
  lower cost — both blocked by the feasibility gate.
- Distractor confusion: cancelled orders, out-of-service vehicles, stale traffic.

## Feasibility guarantee
Every instance is feasible by construction, and remains feasible after each
disruption: total demand is bounded so a solution exists on the remaining fleet
after a breakdown, and time windows are set from a global horizon so a
capacity-feasible plan is always time-feasible.

## Limitations
- Time windows are loose in v1 (capacity and recovery are the binding constraints);
  binding windows are planned for a later version.
- The reference is a construction heuristic, not a proven optimum, so `task` is a
  ratio against a strong baseline rather than a true optimality gap on large instances.
- Single problem family (vehicle dispatch); scheduling and packing members are planned.
