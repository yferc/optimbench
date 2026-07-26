# Contributing to OptimBench

Thanks for your interest. OptimBench is small and opinionated on purpose, so a little
setup goes a long way.

## Development setup

The workflow is uv-based:

```bash
uv venv
uv pip install -e ".[dev,rl]"
pre-commit install
```

The `rl` extra pulls in torch, which is only needed to train or run the learned agent.
The rendering dependencies (imageio, pygame) are base dependencies, so a bare install can
already run every script.

## The gate

Before you open a pull request, both of these must be clean. `pre-commit install` wires
them as a commit hook, and CI runs the same commands:

```bash
ruff check optimbench tests scripts
SDL_VIDEODRIVER=dummy pytest -q
```

`SDL_VIDEODRIVER=dummy` runs the pygame renderer headless, which is what CI does.

## Conventions

The full engineering standard lives in [CLAUDE.md](CLAUDE.md). The rules that reviewers
will hold a change to:

- Fail fast. Internal code assumes valid inputs and raises loudly on a broken assumption.
  No defensive `try/except`, no `dict.get(...)` with a default to paper over a missing key.
  The only sanctioned `try/except` are true system boundaries (the LLM network client, and
  decoding an untrusted LLM reply).
- No magic strings. Every string that carries meaning is an enum (see `optimbench/domain/enums.py`).
- Absolute imports only, no imports inside functions (the one exception is the lazy torch
  import behind the optional `rl` extra).
- Logging, never `print()`. Comments are rare and earn their place; prefer a clearer name.
- No em-dashes anywhere in the repo. Human punctuation only.
- The layering rule: every layer imports only from `optimbench.domain`. The verifier never
  imports the simulation. Only `evaluation/` and `scripts/` may wire several layers together.

## Determinism

The domain rules, the reference solver, and the verifier are fully deterministic, and
scenario generation is seeded. Do not introduce unseeded randomness into these paths, and
respect the train/validation/test seed split in `optimbench/evaluation/splits.py`.

## Adding a new agent

Implement the `Agent` protocol in `optimbench/agents/base.py` (a `reset()` and an `act()`
that maps an observation to an action plus its arguments). Nothing else needs to change to
benchmark it.
