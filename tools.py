import wikipedia
from langchain.tools import BaseTool

class WikipediaTool(BaseTool):
    name: str = "wikipedia"
    description: str = "Useful for answering questions by searching Wikipedia."

    def _run(self, query: str) -> str:
        try:
            summary = wikipedia.summary(query, sentences=3)
            return summary
        except Exception as e:
            return f"Error fetching Wikipedia summary: {e}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

wiki_tool = WikipediaTool()