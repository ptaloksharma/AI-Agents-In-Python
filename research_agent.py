from langchain.llms.base import LLM
from transformers import pipeline
from tools import search_tool, wiki_tool, save_tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from pydantic import PrivateAttr

# Subclass Tool to truncate tool outputs for brevity and compatibility
class TruncatedTool(Tool):
    is_single_input: bool  # declare attribute for pydantic

    def __init__(self, tool, max_chars=300):
        super().__init__(name=tool.name, description=tool.description)
        self._original_tool = tool
        self._max_chars = max_chars
        # Copy is_single_input attribute from original tool for validation
        self.is_single_input = getattr(tool, "is_single_input", True)

    def _run(self, query: str):
        result = self._original_tool.run(query)
        if len(result) > self._max_chars:
            return result[:self._max_chars] + "..."
        return result

    async def _arun(self, query: str):
        return self._run(query)

# Custom HuggingFace LLM wrapper implementing bind_tools and _llm_type
class CustomHuggingFaceLLM(LLM):
    _pipeline: object = PrivateAttr()
    _tools: list = PrivateAttr(default_factory=list)

    def __init__(self, pipeline, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = pipeline

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"

    def _call(self, prompt, stop=None):
        outputs = self._pipeline(prompt, max_length=512)
        return outputs[0]['generated_text']

    @property
    def _identifying_params(self):
        return {"model": str(self._pipeline.model)}

    def bind_tools(self, tools):
        self._tools = tools
        return self

# Initialize HuggingFace pipeline with max_length=512 to avoid overflow
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # smaller model to fit free Colab GPU
    device=0,
    max_length=512
)

llm = CustomHuggingFaceLLM(generator)

# Wrap your existing tools with TruncatedTool to limit output length
tools = [
    TruncatedTool(search_tool, max_chars=300),
    TruncatedTool(wiki_tool, max_chars=300),
    save_tool  # assuming save_tool is already a proper Tool
]

# Concise system prompt instructing the model on expected format
system_message = """
You are a research assistant. Follow this format exactly:

Thought: your reasoning
Action: one of [search, wikipedia, save]
Action Input: input for the action

Final Answer: your final answer

Keep responses concise and to the point.
"""

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    system_message=system_message
)

query = input("What can I help you research? ")

response = agent_executor.run(query)
print("Agent output:")
print(response)