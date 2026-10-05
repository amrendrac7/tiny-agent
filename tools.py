from typing import Callable, Any
import json
from schema import Response


def multiply(a: str, b: str) -> float:
    return float(a) * float(b)


class Tools:
    """Tool registry for the Agent."""

    def __init__(self, requires_approval: list[str] = []):
        """Initialize and select tools that require approval before execution."""
        self.registry = {}
        self.registry_approval = requires_approval

    def observation(self, result: str) -> tuple[str, str]:
        """Return the observation as a user."""
        return "user", f"OBSERVATION: {result}"

    def is_done(self, response: Response) -> bool:
        """The TinyAgent's stopping mechanism."""

        if not response.tool_call:
            return True
        if  response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs", "")
            return True
        return False

    def parse(self, response: Response) -> Response:
        """Parse a JSON tool call from text."""
        text = response.content
        if '"tool":' in text or '"tool:"' in text:
            start, end = text.find("{"), text.rfind("}") + 1
            tool_call = json.loads(text[start: end])

            # add the parsed tool call in the resposne
            return Response(
                content=response.content,
                reasoning=response.reasoning,
                tool_call=tool_call
            )
        return response

    def execute(self, response: Response) -> Any:
        """Run a resistered tool.
        
        Arguments:
            tool_call: A parsed tool call dict with "tool" and "kwargs" keys.
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # human-in-loop: ask before running dangerous tools
        if name in self.registry and name in self.registry_approval:
            response = input(f"Allow {name}? [y/N]").strip().lower()
            if response not in ("y", "yes"):
                return f"Tool '{name}' was denied by the user."

        # handle registered tools
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            print(**kwargs)
            return tool_func(**kwargs)

        return f"Tool '{name}' not found."

    def add_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the Agent can use.

        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
        """
        self.registry[name] = {"function": func, "description": description}

    @property
    def schemas(self) -> None:
        """Used only for native tool-calling."""
        return None

    @property
    def descriptions(self) -> str:
        """Get descriptions of all registered tools."""
        return "\n".join(
            f"`{tool}`: {self.registry[tool]['description']}"
            for tool in self.registry
        )

    @property
    def propmt(self) -> str:
        return f"""
# Tools

If needed, you can use the following tools to assist you
in completing tasks:

{self.descriptions}

To use a tool, respond with JSON:
{{ "tool: "name", "kwargs: {{"params": "value"}}}}
"""


def test_tools():
    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)"
    )

    response = Response(
        content='{"tool": "multiply", "kwargs": {"a": "3.1", "b": "6.5"}}'
    )
    print(tools.schemas)
    print(tools.propmt)
    response = tools.parse(response)

    print(tools.execute(response))


if __name__ == "__main__":
    test_tools()