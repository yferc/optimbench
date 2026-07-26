# CLAUDE.md

This file guides Claude Code sessions working in the OptimBench repo. Follow these conventions exactly.

## Project overview

OptimBench is a verifiable reinforcement-learning environment for dynamic vehicle dispatch. The problem is a capacitated vehicle routing task with time windows, plus disruptions that hit mid-episode (a breakdown that removes the busiest vehicle, a rush order, or a cancellation). The agent assigns orders, sequences routes, and commits a plan through a tool API, then has to recover to a valid, low-cost plan after each disruption.

The reward is a deterministic, reward-hacking-resistant terminal signal (RLVR style): no model-in-the-loop judge. Three scores are computed by code: `task` (dispatch quality behind a hard feasibility gate, capped at 1 against an independent reference solve), `robustness` (fraction of post-disruption states left feasible), and `integrity` (whether the result was reached honestly).

Three baseline agents exist: a greedy heuristic, a learned REINFORCE policy (the optional torch path), and an LLM ReAct agent that runs against any OpenAI-compatible endpoint.

## Build / test / run

The workflow is uv-based. Create the environment and install the extras you need:

```bash
uv venv
uv pip install -e ".[dev,media,rl]"
```

The extras are: `dev` (pytest and ruff, ruff pinned to 0.16.0), `media` (imageio, imageio-ffmpeg, pygame for rendering), `rl` (torch, optional), and `solver` (ortools). Install only what a task needs.

Run the tests headless. pygame requires a video driver, so set `SDL_VIDEODRIVER=dummy` (this is exactly what CI does):

```bash
SDL_VIDEODRIVER=dummy uv run pytest -q
```

Lint with ruff (line length is 100, configured in pyproject.toml). CI runs `ruff check optimbench`:

```bash
uv run ruff check optimbench
```

Scripts:

```bash
uv run python scripts/run_episode.py --agent random   # roll out one episode, export GIF + MP4
uv run python scripts/benchmark.py                     # print the agent comparison table
uv run python scripts/train_rl.py --episodes 3000      # train the policy, writes models/assignment_policy.pt
uv run python scripts/run_llm.py --difficulty easy     # evaluate an LLM agent through the tool API
```

`run_episode.py --agent learned` needs the `rl` extra and a trained model at `models/assignment_policy.pt`. The `llm` agent needs the `OPTIMBENCH_LLM_BASE_URL`, `OPTIMBENCH_LLM_API_KEY`, and `OPTIMBENCH_LLM_MODEL` environment variables (see the docstring in `scripts/run_llm.py`).

## Architecture

The package is layered and infrastructure-first. A new constrained-optimization problem can be added without touching the framework.

```
optimbench/
  domain/        value objects, enums, feasibility rules, schedule, the reference solver (depends on nothing but numpy)
  generation/    procedural, feasibility-guaranteed scenario generation
  simulation/    the environment and the agent tool API
  verification/  the deterministic verifier (feasibility, objective, integrity)
  agents/        the Agent protocol, greedy baseline, learned policy, LLM agent
  evaluation/    metrics and an evaluator with IQM and bootstrap confidence intervals
  rendering/     a top-down renderer and GIF/MP4 export
```

The one hard rule: every layer imports only from `optimbench.domain`. Layers do not import each other. In particular the verifier never imports the simulation, it inspects the final state and trajectory only. `domain` itself depends on nothing but numpy. Keep it that way.

## Coding conventions

This is the most important section. Follow every rule.

- Absolute imports only. Always `from optimbench.x import ...`. Never relative imports (`from .` or `from ..`).
- No imports inside functions, with one justified exception: `torch` is imported lazily inside the learned-agent factory (`_learned_agent` in `scripts/run_episode.py` and the equivalent branch in `scripts/benchmark.py`) because torch is the optional `rl` dependency and the core package must import without it. Do not add other in-function imports.
- Use enums, never magic-string comparisons or string branches. The existing enums are `AgentType`, `OrderFilter`, `Priority`, `Difficulty`, `ActionType`, `DisruptionType`, `ViolationType`, `IntegrityFlag`, and `OrderStatus`. Reach for one of these, or add an enum, instead of comparing raw strings.
- Never use em-dashes anywhere in the repo: code, docstrings, comments, docs, commit messages. Use commas, colons, periods, or parentheses. Human punctuation only.
- Use logging, never `print()`, in both scripts and library code. Get the logger with `logging.getLogger("optimbench")`.
- Argparse defaults live in `add_argument(default=...)`. Do not implement defaults with `x or fallback`.
- Clean style: small functions, self-explanatory names, type hints on parameters and return types, comments extremely rare (the code should read without them), prefer vectorized numpy, least coupling between modules, avoid deep nesting.
- torch is optional (the `rl` extra). The core package and the default test run must work without torch. Tests that need it call `pytest.importorskip("torch")` at module top so they skip cleanly when torch is absent. Keep new torch-dependent code out of import paths that the core package or non-rl tests exercise.
- Determinism: the domain rules, the reference solver, and the verifier are fully deterministic. Scenario generation is seeded. Preserve this. Do not introduce unseeded randomness into these paths.

## Notes

- `models/` holds trained policies; `docs/media/` holds rendered episodes. Caches and build artifacts are git-ignored (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`, `build/`, `dist/`).
- The benchmark task definition, failure modes, and limitations live in `papers/benchmark_card.md`. Consult it before changing scoring, the feasibility gate, or the disruption model.
