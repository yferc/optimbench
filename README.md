<div align="center">

# 🧭 OptimBench

**Verifiable environments for evaluating optimization & planning agents under real-world constraints.**

</div>

---

Modern AI agents can *reason* — but knowing whether they can **reliably operate under constraints** (limited resources, hard deadlines, shifting conditions) is still hard to measure. Task-completion scores hide *how* an agent fails, and most benchmarks can't tell a genuinely-good solution from one that gamed the grader.

**OptimBench** is infrastructure for building and evaluating those agents in the one domain where correctness is *checkable*: **optimization.** Constraints are explicit, objectives are measurable, invalid solutions are detectable, and optimality gaps can be estimated — so the reward is deterministic and hard to fake.

### What it gives you
- 🧩 **Clean, extensible abstractions** — add a new constrained-optimization problem without rewriting the framework (`ScenarioGenerator · Environment · Verifier · Agent · Evaluator`).
- ✅ **A verifier-first design** — the crown jewel is deterministic verification, not the simulator.
- 📊 **Three scores, not one** — `task` (optimization quality behind a feasibility gate), `robustness` (recovery after disruptions), and **`integrity`** (did the agent avoid gaming the verifier?).
- 🧾 **Trajectory logging** — every decision, rationale, and failure, because evaluation is about understanding *why* agents fail.
- 🎲 **Procedural generation** — seeded, feasibility-guaranteed instances; impossible to memorize, difficulty scalable.

### Status
🚧 **v0 — scaffolding.** First environment: a minimal *dynamic dispatch* problem (assign & route a small fleet, then recover from a disruption), with greedy / OR-Tools / LLM baselines and a benchmark card. See [`papers/benchmark_card.md`](papers/benchmark_card.md).

## License
MIT
