from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

result = search.invoke(
    "What are the latest developments in AI agents?"
)

print(result)