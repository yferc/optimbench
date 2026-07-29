# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The benchmark scoring itself
is versioned separately by `BENCHMARK_VERSION` (see `optimbench/domain/version.py`).

## [Unreleased]

### Changed
- Benchmark v1.1: the REROUTE auto-router now sequences stops on the observed (possibly stale)
  travel times the agent can see, not the true times, so it no longer hands the agent an oracle
  the stale-traffic dimension is meant to deny it. Scores shift slightly down and the effect
  grows with difficulty (where more of the traffic matrix is stale). The reference solve still
  uses true times, since it is the agent-independent oracle denominator. BENCHMARK_VERSION bumped
  to 1.1, so v1.0 and v1.1 numbers are not comparable.

### Added
- `scripts/hub_baseline.py`: score greedy / learned / random on the *hub* reward (integrity gate
  times weighted task and robustness) over the same TEST_SEEDS and difficulty a `prime eval run`
  uses. The repo leaderboard reports task score per difficulty, which is a different scale, so
  comparing a hub LLM number against it was an apples-to-oranges error waiting to happen. Now
  every published row sits on one metric.
- A measured zero-shot LLM table in the README and the hub card, on the hub reward at `medium`:
  claude-opus-5 0.917, learned 0.870, deepseek-v3.2 0.766, greedy 0.762, gemini-2.5-flash-lite
  0.547, gpt-4.1-mini 0.273, gpt-5-nano 0.000, random 0.000. A frontier model now clears both the
  heuristic and the trained policy zero-shot, so the earlier claim that LLMs do not clear this bar
  is retired; the separation at the middle and bottom of the range, and the recovery-after-disruption
  failure mode, are what the benchmark actually demonstrates. Sample size (3 seeds) and sampling
  settings are stated inline so the pilot is not mistaken for the 20x3 reproducible protocol, and
  pre-feedback LLM figures are explicitly marked as not comparable.
- Offline optimality-gap analysis (`optimbench/analysis/`, `scripts/optimality_report.py`, the
  `solver` extra): an OR-Tools CVRP optimum used only to report how tight the heuristic reference
  (the task-score denominator) is. Measured gap: the reference sits about 14% above optimal on
  easy and medium, 19% on hard. Kept strictly out of the scoring path, so scoring stays
  deterministic and solver-free.
- `scripts/replay.py`: replay one episode as a readable narrative (each mutation and commit, when
  a disruption hits, which wave went infeasible, and the final verdict with the integrity flag that
  tripped), for any agent. The qualitative companion to the leaderboard.
- `scripts/export_trajectories.py`: export expert dispatch trajectories (greedy by default) as
  JSONL (prompt, completion) pairs from successful episodes, for supervised warm-start before RL.
- A Prime Intellect hub / verifiers adapter (`optimbench/hub/`, the `hub` extra, and the
  `environments/optimbench_dispatch/` package) exposing OptimBench behind `load_environment()`
  so it runs under `prime eval run` and prime-rl. The deterministic verifier maps to a verifiers
  Rubric reward, with the three scores logged as zero-weight metrics. Verified against
  verifiers 0.1.14 and 0.2.1; the hub package pins verifiers>=0.2.1 (Python 3.11+).
- `combined_reward`: one scalar RL signal in [0, 1] with integrity as a hard multiplicative gate.
- A reward-integrity audit suite (`tests/test_reward_integrity.py`) proving the gate and flags
  fire on adversarial trajectories and that shortcuts score zero.
- Type information for downstream consumers: a `py.typed` marker (PEP 561).
- A curated top-level public API in `optimbench` with an explicit `__all__`.
- Docstrings on the public extension points (the `Agent` protocol, `DispatchEnvironment`,
  `DispatchVerifier`, `Evaluator.evaluate`) and module docstrings across the package.
- Disjoint train / validation / test seed splits (`optimbench/evaluation/splits.py`), so
  checkpoint selection no longer peeks at the reported test set.
- `BENCHMARK_VERSION`, stamped into `EvaluationReport` and printed by the benchmark script.
- Separate `terminated` (final commit) and `truncated` (per-wave turn cap) episode endings,
  exposed as environment properties and in the step result.
- `CONTRIBUTING.md`, `CITATION.cff`, and this changelog.

### Changed
- Rejected and noted actions now report back to the agent. The environment already computed an
  explanatory note for every outcome (out-of-service vehicle, unknown id, node out of range,
  malformed call, disruption applied), but both the local runner and the hub adapter discarded it
  and re-rendered only the state, so an LLM agent could not tell an accepted action from a rejected
  one. The observation now carries the last action's outcome and `render_state` surfaces it as a
  salient line, so a model stops blindly repeating an invalid call. This is agent feedback only:
  the deterministic scoring, feasibility rules, reference solve, and BENCHMARK_VERSION are
  unchanged; the greedy, learned, and reference agents are unaffected.
- The LLM system prompt now derives its example tool call from the enums, so the schema the
  model is told to emit cannot drift from the keys the parser reads.
- The `Agent` protocol and LLM agent are typed with the `Field` and `Arg` enums, matching
  every other agent and the environment; the LLM decode boundary re-keys parsed arguments
  to the `Arg` enum.

### Removed
- The unused `solver` (ortools) optional dependency.

## [0.0.1]
- Initial release: the dynamic vehicle dispatch environment, the deterministic three-score
  verifier, greedy / learned / LLM / random agents, seeded generation, IQM and bootstrap-CI
  evaluation, and a top-down GIF/MP4 renderer.
