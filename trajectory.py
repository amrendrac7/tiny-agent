from dataclasses import dataclass
from llm import Response


@dataclass
class Step:
    """A single step in an agent's trjectory."""

    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent executon as a sequence of runs."""
    def __init__(self):
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        # add thought
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # add action/observation or answer
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)