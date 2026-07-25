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
- **task** — `min(1, reference_cost / committed_cost)` behind a hard feasibility gate (0 if infeasible). `reference_cost` is a strong deterministic solver (nearest-neighbour + 2-opt) run on the final committed state, so the score reads as routing efficiency.
- **robustness** — fraction of committed states that were feasible, counting every disruption wave the agent was expected to face; waves left unfaced score as failures.
- **integrity** — 1.0 unless the agent spammed invalid actions, never committed, or ended with disruptions unresolved.

Feasibility (the gate) enforces: capacity, full live-order coverage, depot-anchored
routes, and shift limits on a service-time-aware schedule. Time windows are modeled
but generously bounded in v1 (see Limitations).

## Baselines
Greedy dispatcher (best-fit assignment + nearest-neighbour routing, re-planning
after each disruption), 50 scenarios per difficulty: feasibility 100% across
easy/medium/hard; task 0.98 / 0.96 / 0.93 (easy/medium/hard); robustness 1.00;
integrity 1.00. Greedy is always feasible and resolves every disruption, so it
sets an honest task floor that drops with difficulty as nearest-neighbour routing
falls further behind the 2-opt reference.

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
- Time windows are loose in v1 (capacity, coverage, depot-anchoring and recovery are
  the binding constraints); binding windows that preserve the feasibility guarantee
  are planned for a later version.
- The reference is a nearest-neighbour + 2-opt heuristic, not a proven optimum, so
  `task` measures routing efficiency against a strong baseline rather than a true
  optimality gap. It is computed on the agent's own assignment, so it rewards good
  routing given an assignment rather than optimal fleet balancing.
- Single problem family (vehicle dispatch); scheduling and packing members are planned.
