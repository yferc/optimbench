# OptimBench — Benchmark Card

> Treat this like a mini-paper. Filled in as v1 lands.

## Motivation
Modern agents can produce answers; knowing whether they operate **reliably under
constraints** is unsolved. Optimization gives us a domain where correctness is
*checkable*: constraints are explicit, objectives are measurable, invalid
solutions are detectable, and optimality gaps can be estimated.

## Task definition
_What the agent observes and controls (per environment)._ — TBD

## Evaluation
Three deterministic scores, no LLM judge:
- **task** — optimization quality behind a hard feasibility gate (reference / achieved cost, capped at 1).
- **robustness** — fraction of post-disruption states left feasible.
- **integrity** — did the agent solve it *without* gaming the verifier (invalid actions, exploiting simulator bugs, reward hacking)?

## Baselines (the performance floor)
- Greedy heuristic — TBD
- OR-Tools / exact reference — TBD
- Naive LLM agent — TBD
- (optional) Human operator — TBD

## Failure modes
_Which strategies fail and why (e.g. premature "done" before repairing a post-disruption violation)._ — TBD

## Limitations
_What this does NOT measure._ — TBD
