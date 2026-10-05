from llm import LLM, EmbeddingModel


class Memory:
    """Simple memory to store conversation history"""

    def __init__(self):
        self.messages = []

    def add(
        self,
        role: str,
        content: str,
        tool_call: dict | None = None,
        **kwargs
    ) -> None:
        """Add a message to memory"""
        message = {"role": role, "content": content}

        # tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # append message to memory
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Get all messages"""
        return self.messages


class TrimmingMemory(Memory):
    """Memory that keeps only the last two user/assistant turns"""
    def add(self, role: str, content: str, **kwargs) -> None:
        # add the new message using the parent class
        super().add(role, content, **kwargs)
        system = [
            message for message in self.messages if message[role] == "system"
        ]
        truns = [
            message for message in self.messages if message[role] != "system"
        ]
        
        self. messages = system  + truns[-4:]


class SummarizationMemory(Memory):
    """Memory that compress past turns into a running summary"""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm

    def add(
            self,
            role: str,
            content: str,
            **kwargs
    ) -> None:
        # add the new message first using the parenr class
        super().add(role, content, **kwargs)

        # after each cimpleted turn, update the running summary
        if role == "assistant":
            # split the existing summary from the new conversation
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary = message["content"]
                else:
                    conversation += f"{message['role']}: {message['content']}\n"

            prompt = f"""Update the summary with the new conversation.

Summary: {summary}

Conversation:
{conversation}
Output the updated summary only."""
            response = self.llm.generate([{"role": "user", "content": prompt}])
            self.messages = [{"role": "system", "content": response.content}]


class RAGMemory(Memory):
    """Long-term Memory with RAG."""

    def __init__(
            self,
            embedding_model: EmbeddingModel,
            documents: list[str],
        ):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [embedding_model.embed(doc) for doc in documents]

    def add(
            self, 
            role: str,
            content: str,
            **kwargs
    ):
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
{context}

Question: {content}"""
        
        super().add(role, content)

    def search(self, query: str) -> list[str]:
        """Return the top-k documents most similar to the query"""
        query_emebd = self.embedding_model.embed(query)
        scores = [self._cosine(query_emebd, embed) for embed in self.embeddings]
        ranked = sorted(
            range(len(scores)), key=lambda i:scores[i], reverse=True
        )

        return [self.documents[index] for index in ranked[:3]]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / norm
