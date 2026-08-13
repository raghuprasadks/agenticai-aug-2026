from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import requests
from urllib.parse import quote


wikipedia_api = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=wikipedia_api
)


def fetch_summary_fallback(query: str) -> str:
    """Fallback to Wikipedia REST API when tool JSON parsing fails."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
    headers = {
        "User-Agent": "langchaindemo/1.0 (educational script)"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    data = response.json()

    title = data.get("title", query)
    summary = data.get("extract", "No summary found.")
    return f"Page: {title}\nSummary: {summary}"


def query_wikipedia(query: str) -> str:
    try:
        return wikipedia_tool.invoke(query)
    except requests.exceptions.JSONDecodeError:
        # Some network/proxy paths return HTML or empty text instead of JSON.
        return fetch_summary_fallback(query)
    except requests.exceptions.RequestException as exc:
        return f"Network error while calling Wikipedia: {exc}"
    except Exception as exc:
        message = str(exc)
        if "JSONDecodeError" in message or "Expecting value" in message:
            return fetch_summary_fallback(query)
        raise


result = query_wikipedia("Alan Turing")

print(result)