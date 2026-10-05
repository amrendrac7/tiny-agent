import json
import urllib.request
from schema import Response


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
    ):
        """Initialize the LLM with given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think

    def generate(
        self,
        messages: list[dict],
        tools: list | None = None
    ) -> Response:
        """Generate a resposne from the LLM given a list of messages and tools"""
        # Build the rerquest body
        body = {
            "model": self.model,
            "messages": messages,
        }
        # tools and reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request) as resposne:
            data = json.loads(resposne.read())

        # Extract message, tool_call and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_call[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"]
        }

        # format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata,
        )


class EmbeddingModel:
    """Generate embedding"""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
    ):
        """Initialize the embedding model with the given model."""
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        # POST to OpenAI-compatible /embeddings endpoint
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": text}).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request) as resp:
            resposne = json.loads(resp.read())

        # extract and run the embedding
        return resposne["data"][0]["embedding"]