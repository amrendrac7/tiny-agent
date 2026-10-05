from llm import LLM, EmbeddingModel
from trajectory import Trajectory
from memory import SummarizationMemory, Memory, RAGMemory
from tools import Tools
from schema import Response


class TinyAgent:
    """A minimal, modular, and educational agent framework"""

    def __init__(
            self,
            llm: LLM,
            memory: Memory,
            tools: Tools,
        ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None

        self.trajectory = Trajectory()

        # build the system prompt with all components
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.tools.propmt
        self.memory.add("system", system_prompt)

    def run(self, task: str) -> str:
        """Run the agent on a task"""
        self.memory.add("user", task)
        self.trajectory.initialize(task)

        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single task"""
        # Thought: Generate response and add to memory
        response = self.llm.generate(
            self.memory.get_messages(),
            tools=self.tools.schemas
        )
        print("response", response)
        self.memory.add(
            "assistant", response.content,
            tool_call=response.tool_call
        )
        # Tool parsing
        response = self.tools.parse(response)

        #answer: stopping mechanism
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)

    def _execute_action(self, response: Response) -> str:
        """Execute a tool action"""
        # Action: execute tools
        result = self.tools.execute(response)

        # observation: add results to memory and display
        role, observation = self.tools.observation(result)
        self.memory.add(role, observation)
        self.trajectory.add(role, observation)
        
        return observation


def test_llm():
    llm = LLM(model="gemma4:e2b")

    response = llm.generate(
        [{"role": "user", "content": "Hi!, How's life"}]
    )
    print(response)


def test_embedding():
    embedding_model = EmbeddingModel(model="embeddinggemma")

    output = embedding_model.embed("Dolphins are amazing!")
    # print(output)
    print(len(output))


def rag_agent():
    embedding_model = EmbeddingModel(model="embeddinggemma")
    llm = LLM(model="gemma4:e2b")

    # RAGMemory with external documents
    documents = [
        "Sarah works as a marine biologist studying coral reefs.",
        "Sarah lives in Lisbon, Portugal.",
        "Sarah's favorite hobby is rock climbing.",
        "Sarah favorite animal is flamingos.",
        "Sarah speaks fluent Spanish and Portuguese.",
        "Ilse is a software engineer at a renewable energy startup.",
        "Ilse lives in Amsterdam, the Netherlands.",
        "Ilse plays the cello in a local string quartet.",
        "Ilse's favorite author is Brandon Sanderson.",
        "Ilse's favorite animal is dolphins.",
    ]
    memory = RAGMemory(documents=documents, embedding_model=embedding_model)
    agent = TinyAgent(llm=llm, memory=memory)
    response = agent.run("What is Sarah's favourite animal?")
    print(response)
    print(agent.memory.get_messages())

def main():
    llm = LLM(model="gemma4:e2b")
    memory = SummarizationMemory(llm)

    agent = TinyAgent(llm, memory)

    response = agent.run("Hi! my name is Sarah. Wny are flamingos pink?")
    print(agent.memory.get_messages())
    # print(agent.trajectory.runs)


def multiply(a: str, b: str) -> float:
    return float(a) * float(b)


def test_tooling_agent():
    llm = LLM(model="gemma4:e2b")

    # register tool
    tools = Tools()
    tools.add_tool(
        name="multiply",
        func=multiply,
        description="Multiplies two numbers: multiply(a: str, b: str)"
    )

    memory = Memory()

    agent = TinyAgent(
        llm=llm,
        memory=memory,
        tools=tools,
    )

    agent.run("What is 5.1 times 7.3?")


if __name__ == "__main__":
    # test_llm()
    # rag_agent()
    test_tooling_agent()
    # test_tools()
    # main()


