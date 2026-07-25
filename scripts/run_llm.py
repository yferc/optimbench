"""Evaluate an LLM agent through the tool API and print the metric report.

Point it at any OpenAI-compatible endpoint via environment variables:

    # Groq (free tier)
    export OPTIMBENCH_LLM_BASE_URL=https://api.groq.com/openai/v1
    export OPTIMBENCH_LLM_API_KEY=gsk_...
    export OPTIMBENCH_LLM_MODEL=llama-3.3-70b-versatile

    # Google Gemini (free tier)
    export OPTIMBENCH_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    export OPTIMBENCH_LLM_API_KEY=...
    export OPTIMBENCH_LLM_MODEL=gemini-2.0-flash

    # Ollama (local)
    export OPTIMBENCH_LLM_BASE_URL=http://localhost:11434/v1
    export OPTIMBENCH_LLM_MODEL=qwen2.5:7b

    python scripts/run_llm.py --difficulty easy --seeds 5

Each episode makes one API call per turn, so start with a few seeds.
"""

from __future__ import annotations

import argparse

from optimbench.agents import openai_compatible_agent
from optimbench.domain import Difficulty
from optimbench.evaluation import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="easy", choices=[d.value for d in Difficulty])
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    report = Evaluator().evaluate(
        openai_compatible_agent, Difficulty(args.difficulty), range(args.seeds)
    )
    print(report.format())


if __name__ == "__main__":
    main()
