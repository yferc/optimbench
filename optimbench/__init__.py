"""OptimBench: verifiable environments for evaluating optimization and planning agents.

This top level re-exports the supported public API. Importing from submodules
still works, but only the names below carry stability guarantees across versions.
"""
from optimbench.agents import (
    Agent,
    AgentType,
    GreedyDispatcher,
    LLMAgent,
    OpenAICompatibleClient,
    RandomDispatcher,
    openai_compatible_agent,
)
from optimbench.domain import Difficulty
from optimbench.evaluation import EvaluationReport, Evaluator
from optimbench.generation import DispatchScenarioGenerator
from optimbench.simulation import DispatchEnvironment
from optimbench.verification import DispatchVerifier, VerificationResult

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "AgentType",
    "Difficulty",
    "DispatchEnvironment",
    "DispatchScenarioGenerator",
    "DispatchVerifier",
    "EvaluationReport",
    "Evaluator",
    "GreedyDispatcher",
    "LLMAgent",
    "OpenAICompatibleClient",
    "RandomDispatcher",
    "VerificationResult",
    "__version__",
    "openai_compatible_agent",
]
