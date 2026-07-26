# optimbench (Prime Intellect environment)

Dynamic vehicle dispatch as a verifiable, multi-turn RL environment: a capacitated routing
problem with time windows and mid-episode disruptions (a breakdown of the busiest vehicle, a
rush order, a cancellation). The agent assigns orders, sequences routes, and commits a plan
wave by wave through a ten-tool API, recovering after each disruption.

This is the Prime Intellect hub wrapper around the standalone
[optimbench](https://github.com/yferc/optimbench) library. The library holds the environment,
the deterministic verifier, and the baselines; this package exposes them behind the verifiers
`load_environment()` contract so they run under `prime eval run` and prime-rl unchanged.

## Reward

A single scalar in `[0, 1]`, computed by code with no model-in-the-loop judge (RLVR):

```
reward = integrity_gate * (0.7 * task + 0.3 * robustness)
```

Integrity is a hard gate: an episode that never commits, leaves a disruption unresolved, or
spams invalid actions scores zero regardless of dispatch quality. Behind the gate, `task` is
the cost ratio to an agent-independent reference solve (capped at 1, zero if infeasible) and
`robustness` is the fraction of post-disruption waves left feasible. The three scores are also
logged as zero-weight metrics so `prime eval view` shows the breakdown.

## Arguments

`load_environment(difficulty="medium", max_turns_per_wave=80, train_size=128)`. `difficulty`
is `easy`, `medium`, or `hard`. Evaluation uses the held-out `TEST_SEEDS`; training uses
disjoint `TRAIN_SEEDS`.

## Required environment variables

None. The reward is fully deterministic and local.

## Install and evaluate

The wrapper depends on the `optimbench` library. Until it is on PyPI, install the library
first (editable), then the environment:

```bash
pip install -e /path/to/optimbench          # the core library
prime env install optimbench                # this wrapper
prime eval run optimbench -m openai/gpt-5-nano
prime eval view
```

Publishing (`prime env push`) is done from your own Prime Intellect account.
