# OptimBench

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![Reward](https://img.shields.io/badge/reward-deterministic%20(RLVR)-orange)

Verifiable environments for evaluating optimization and planning agents under real-world constraints.

Modern language-model agents can reason, but it is still hard to measure whether they can *reliably operate under constraints*: limited resources, hard deadlines, and conditions that change mid-task. Task-completion scores hide how an agent fails, and most benchmarks cannot tell a genuinely good solution from one that quietly gamed the grader.

OptimBench is built around the one setting where correctness is checkable by code: optimization. Constraints are explicit, objectives are measurable, invalid solutions are detectable, and the gap to a reference is quantifiable. That makes the reward deterministic and hard to fake.

<p align="center">
  <img src="docs/media/demo.gif" width="520" alt="A dispatch agent assigning orders, routing vehicles, and recovering from a disruption">
</p>

The first environment is dynamic vehicle dispatch. An agent is given a fleet, a set of delivery orders with capacities and time windows, and a road network. It assigns orders, sequences routes, and commits a plan, and then a disruption hits (the busiest vehicle breaks down, an urgent order arrives, an order cancels). The agent has to recover to a valid, low-cost plan. Because the disruptions cascade and the breakdown always takes out whichever vehicle the agent leaned on most, a single-shot answer is impossible by construction.

## What it measures

Every score is computed deterministically, with no model-in-the-loop judge.

- **task**: dispatch quality behind a hard feasibility gate. The committed cost is compared to a strong deterministic reference that solves the same instance from scratch (sweep assignment, then nearest-neighbour plus 2-opt routing), capped at 1. Because the reference is independent of how the agent assigned its orders, this rewards good assignment *and* routing, not just routing. Zero if the final plan is infeasible.
- **robustness**: the fraction of post-disruption states the agent left feasible. Waves the agent never faced count as failures, so stalling to skip a disruption is penalized here.
- **integrity**: whether the agent reached its result honestly, rather than by spamming invalid actions, never committing, or leaving disruption waves unresolved.

The feasibility gate enforces real constraints: vehicle capacity, full order coverage, depot-anchored routes, and shift limits on a service-time-aware schedule. An agent cannot inflate its score by dropping the depot from a route or leaving orders unassigned. (Time windows are modeled but generously bounded in v1; see the benchmark card.)

## Results

A hand-written heuristic and a trained network, one verifier, 50 procedurally generated scenarios per difficulty. The number is **task** (dispatch quality vs the reference); both stay 100% feasible with robustness and integrity at 1.0.

| agent | easy | medium | hard |
| --- | --- | --- | --- |
| greedy heuristic | 0.834 | 0.747 | 0.658 |
| learned RL policy | **0.921** | **0.852** | **0.808** |

- **Greedy**: best-fit assignment plus nearest-neighbour routing, re-planning after each disruption. Always feasible, but its non-geometric assignment leaves real headroom against the sweep-plus-2-opt reference.
- **Learned RL policy**: the same route-then-dispatch skeleton with greedy's assignment replaced by a small policy network scored on order/vehicle geometry, trained with REINFORCE against the greedy baseline. It closes most of that headroom, and the gain grows with difficulty (+0.09 easy → +0.15 hard) because assignment quality dominates on the larger instances.

The point is that this table is produced by code, not judgement: the same deterministic verifier scores a hand-written heuristic and a trained network on equal terms.

### LLM agents

The tool-using agent runs on any OpenAI-compatible endpoint (Groq, Gemini, Ollama, x.ai, and so on) via three env vars (see `scripts/run_llm.py`). As a first probe, a small local model (Qwen2.5-7B through Ollama, ReAct-style with short-term memory) scored **0% feasible on easy over 5 seeds**: it follows the tool protocol and gets the assign, reroute, dispatch shape right, but does not recover cleanly from the breakdown. It leaves an order unassigned and a reloaded vehicle's route stale, then dispatches anyway. The environment is solvable (greedy is 100% feasible on the same seeds); holding feasibility *through a disruption* is the gap this benchmark is built to surface. A hosted frontier model is expected to do far better; the harness is one API key away from running it.

## How it is built

The design is infrastructure first. A new constrained-optimization problem can be added without touching the framework, because every system depends only on the domain layer.

```
domain/        value objects, enums, the feasibility rules, schedule and reference solver
generation/    procedural, feasibility-guaranteed scenario generation
simulation/    the environment and the agent tool API
verification/  the deterministic verifier (feasibility, objective, integrity)
agents/        the Agent interface, a greedy baseline, a learned policy, and an LLM agent
evaluation/    metrics and an evaluator with IQM and bootstrap confidence intervals
rendering/     a top-down renderer and GIF/MP4 export
```

The agent acts through a tool API rather than emitting a single answer: `list_orders`, `query_traffic`, `check_feasibility`, `assign_order`, `set_route`, `reroute`, `dispatch`, and `refuse`.

## Setup (uv + PyCharm)

OptimBench uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. Python 3.10 or newer is required.

```bash
# 1. install uv (macOS)
brew install uv                              # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. clone and create the env (.venv in the project root, with test + rl deps)
git clone https://github.com/yferc/optimbench.git && cd optimbench
uv sync --extra dev --extra rl

# 3. run the tests, the linter, and a sample episode
SDL_VIDEODRIVER=dummy uv run pytest -q        # tests (headless pygame)
uv run ruff check optimbench                  # lint
uv run python scripts/run_episode.py --agent random --difficulty easy
```

Rendering deps (pygame, imageio) are regular dependencies, so scripts run with a bare `uv run`. The optional extras are `dev` (pytest, ruff), `rl` (PyTorch, only to train or run the learned agent), and `solver` (ortools).

**PyCharm:** open the folder, then Settings > Project > Python Interpreter > Add Interpreter > Add Local Interpreter > Existing, and select `.venv/bin/python`. Use that interpreter directly (not the "uv" wrapper), otherwise PyCharm launches through `uv run`, which re-syncs the env and drops optional extras like `rl` mid-debug. Right-click any script under `scripts/` to Run, with the working directory set to the project root.

**Docker:** `docker build -t optimbench .` then `docker run --rm optimbench` (runs the test suite; append a command to run something else).

## Quickstart

```bash
uv run pytest                               # run the tests
uv run python scripts/run_episode.py        # render an episode to docs/media/greedy.gif
uv run python scripts/benchmark.py          # print the agent comparison table
```

Evaluate an agent:

```python
from optimbench.agents import GreedyDispatcher
from optimbench.domain import Difficulty
from optimbench.evaluation import Evaluator

report = Evaluator().evaluate(GreedyDispatcher, Difficulty.MEDIUM, range(50))
print(report.format())
```

Train the learned policy (needs `pip install -e ".[rl]"`), or point the LLM agent at any OpenAI-compatible endpoint:

```bash
python scripts/train_rl.py --episodes 3000        # writes models/assignment_policy.pt

export OPTIMBENCH_LLM_BASE_URL=https://api.groq.com/openai/v1
export OPTIMBENCH_LLM_API_KEY=gsk_...
export OPTIMBENCH_LLM_MODEL=llama-3.3-70b-versatile
python scripts/run_llm.py --difficulty easy --seeds 5
```

## Results

Task score (IQM) on 50 held-out test seeds per difficulty (benchmark v1.1), higher is better.
The reference solve sits at 1.0 by definition. Full table with robustness and integrity in
[`papers/leaderboard.md`](papers/leaderboard.md); reproduce with `python scripts/benchmark.py --seeds 50 --out papers/leaderboard.md`.

| agent   | easy  | medium | hard  |
|---------|-------|--------|-------|
| random  | 0.000 | 0.000  | 0.000 |
| greedy  | 0.825 | 0.739  | 0.648 |
| learned | 0.933 | 0.856  | 0.792 |

Random cannot assemble a feasible plan by chance, so it scores zero on every axis. The learned
REINFORCE policy closes most of the gap greedy leaves against the reference, and the headroom
widens with difficulty.

## Run it as a Prime Intellect environment

OptimBench ships a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) adapter, so it
installs and runs from the Prime Intellect hub and trains under prime-rl with no glue. The
core library stays dependency-light; the adapter lives behind the `hub` extra.

```bash
pip install -e ".[hub]"
prime eval run optimbench -m openai/gpt-5-nano   # from environments/optimbench
```

The deterministic verifier becomes the reward (integrity gate times weighted task and
robustness), the three scores are logged as metrics, and each scenario seed is one dataset
row. See `environments/optimbench/` for the hub package and `optimbench/hub/` for the adapter.

## Roadmap

OptimBench is the first member of a planned family of verifiable optimization environments (scheduling, packing, graph problems) sharing one generator, verifier, and metric suite. See `papers/benchmark_card.md` for the task definition, failure modes, and limitations.

## Citation

```bibtex
@software{ferchichi_optimbench_2026,
  author  = {Ferchichi, Yahia},
  title   = {OptimBench: verifiable environments for evaluating optimization and planning agents},
  year    = {2026},
  version = {0.0.1},
  url     = {https://github.com/yferc/optimbench}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the engineering standard, and
[CLAUDE.md](CLAUDE.md) for the full conventions.

## License

MIT.
