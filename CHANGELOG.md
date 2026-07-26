# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The benchmark scoring itself
is versioned separately by `BENCHMARK_VERSION` (see `optimbench/domain/version.py`).

## [Unreleased]

### Added
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
