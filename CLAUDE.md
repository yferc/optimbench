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
uv pip install -e ".[dev,rl]"
```

The extras are: `dev` (pytest and ruff, ruff pinned to 0.16.0), `rl` (torch, only to train or run the learned agent), and `hub` (verifiers, only for the Prime Intellect hub adapter in `optimbench/hub/`). The rendering deps (imageio, imageio-ffmpeg, pygame) are base dependencies, not an extra, so a bare install can always run the scripts. Install only the extras a task needs.

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
uv run python scripts/export_trajectories.py           # dump expert trajectories as JSONL for SFT
uv run python scripts/replay.py --agent random         # narrate one episode: decisions, disruption, verdict
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

The exceptions are the composition roots: `evaluation/`, `scripts/`, and `hub/` (the verifiers adapter). They exist to wire the system together (generator, environment, verifier, agents) and may import several layers. `hub/` additionally imports the optional `verifiers` dependency, so, like the learned agent's torch import, it must stay out of every path the core package or a non-hub test loads: `optimbench/__init__` must never import it, and its test uses `pytest.importorskip("verifiers")`. Nothing else may cross layers.

## Coding conventions

This is the most important section. Follow every rule.

- Fail fast, do not program defensively. Internal code assumes its inputs are valid and lets a wrong assumption raise loudly. Do not return sentinels or swallow errors to keep running in a degraded state. A clear crash beats a corrupted result that looks correct.
- Aim for zero `try/except`. The only acceptable places are true system boundaries: transient network I/O with explicit retry (the LLM client), and decoding an untrusted LLM response (a malformed reply must not crash a benchmark run). Nowhere else. If you reach for `try/except` to handle a value that "might" be wrong, that value should have been the right type or checked with a plain `if`.
- Never call `dict.get(...)` or `args.get(...)`. Index explicitly (`d[key]`), and where a key's presence is a genuine decision (a game rule, not defensiveness) test it with `in`. A missing key that the contract guarantees should raise, not be papered over with a default.
- No primitive obsession. Every string that carries meaning is an enum, not a literal. This includes tool argument names, result keys, and observation keys, not just comparisons. Existing enums: `AgentType`, `OrderFilter`, `Priority`, `Difficulty`, `ActionType`, `DisruptionType`, `ViolationType`, `IntegrityFlag`, `OrderStatus`, and the tool-schema enums. Add an enum rather than introduce a magic string.
- Prefer explicit, typed parameters over generic `**kwargs` or an untyped `args` dict threaded through functions. A function's signature should name what it needs.
- Absolute imports only. Always `from optimbench.x import ...`. Never relative imports (`from .` or `from ..`).
- No imports inside functions, with one justified exception: `torch` is imported lazily inside the learned-agent factory (`_learned_agent` in `scripts/run_episode.py` and the equivalent branch in `scripts/benchmark.py`) because torch is the optional `rl` dependency and the core package must import without it. Do not add other in-function imports.
- Never use em-dashes anywhere in the repo: code, docstrings, comments, docs, commit messages. Use commas, colons, periods, or parentheses. Human punctuation only.
- Use logging, never `print()`, in both scripts and library code. Get the logger with `logging.getLogger("optimbench")`.
- Argparse defaults live in `add_argument(default=...)`. Do not implement defaults with `x or fallback`.
- Comments are rare and earn their place: only where a name cannot carry the meaning (a non-obvious ratio, an invariant, a formula). Do not comment the obvious. Prefer a clearer name or type over a comment.
- Clean style: small functions, self-explanatory names, type hints on parameters and return types, prefer vectorized numpy, least coupling between modules, avoid deep nesting.
- torch is optional (the `rl` extra). The core package and the default test run must work without torch. Tests that need it call `pytest.importorskip("torch")` at module top so they skip cleanly when torch is absent. Keep new torch-dependent code out of import paths that the core package or non-rl tests exercise.
- Determinism: the domain rules, the reference solver, and the verifier are fully deterministic. Scenario generation is seeded. Preserve this. Do not introduce unseeded randomness into these paths.

## Environment interface

`DispatchEnvironment` is the gym-style loop. `reset(scenario)` returns the first observation; `step(action, args)` takes an `ActionType` and an args dict keyed by the `Arg` enum, and returns `{result, accepted, note, observation}`.

- Observation (a dict): `wave`, `waves_total`, `final_wave`, `feasible`, `depot` coordinates, `vehicles` (id, capacity, in_service, load, assigned, route, load centroid), and `unassigned_orders` (id, node, coord, demand, window, priority).
- Action space: the ten `ActionType` tools in `simulation/tools.py`, each with its `Arg` keys. Read tools return info; mutation tools edit the plan; `dispatch` commits the wave and applies the next disruption.
- Reward is terminal, from the verifier: `task`, `robustness`, `integrity`. See `papers/benchmark_card.md`.

## Working agreements

- Before you commit: `ruff check optimbench tests scripts` is clean, `SDL_VIDEODRIVER=dummy pytest -q` is green, no em-dashes, no new magic strings, no new `try/except` or `dict.get`. `pre-commit install` wires the first two as a hook.
- Commit messages: a short imperative subject, then a body explaining the why. No em-dashes.
- Human-gated: do not mutate datasets, trained checkpoints under `models/`, or experiment logs without explicit sign-off. Regenerate media and models with the scripts, do not hand-edit them.

## Notes

- `models/` holds trained policies; `docs/media/` holds rendered episodes. Caches and build artifacts are git-ignored (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `*.egg-info`, `build/`, `dist/`).
- The benchmark task definition, failure modes, and limitations live in `papers/benchmark_card.md`. Consult it before changing scoring, the feasibility gate, or the disruption model.
