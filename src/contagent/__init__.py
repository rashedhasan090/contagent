"""contagent: Monte Carlo simulation of prompt-injection contagion in multi-agent LLM systems."""

from .scenario import Agent, Edge, Injection, Payload, Scenario, load_scenario, scenario_from_dict
from .engine import Simulation, RunResult
from .montecarlo import Forecast, forecast
from .interventions import apply_intervention, parse_intervention, rank_interventions

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Edge",
    "Injection",
    "Payload",
    "Scenario",
    "load_scenario",
    "scenario_from_dict",
    "Simulation",
    "RunResult",
    "Forecast",
    "forecast",
    "apply_intervention",
    "parse_intervention",
    "rank_interventions",
    "__version__",
]
