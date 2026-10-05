from dataclasses import dataclass


@dataclass
class Response:
    """Structured resposne from LLM calls."""
    
    content: str = ""
    reasoning: str | None = None
    tool_call: str | None = None
    metadata: str | None = None
