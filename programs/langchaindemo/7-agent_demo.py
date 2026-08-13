import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_cohere import ChatCohere

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun
)

from langchain_community.utilities import (
    WikipediaAPIWrapper
)

from langchain_core.tools import tool


load_dotenv()

cohere_api_key = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise RuntimeError("Set CO_API_KEY or COHERE_API_KEY in your .env file.")


# ==========================================
# TOOL 1 : Web Search
# ==========================================

search_tool = DuckDuckGoSearchRun()


# ==========================================
# TOOL 2 : Wikipedia
# ==========================================

wikipedia_api = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=wikipedia_api
)


# ==========================================
# TOOL 3 : Calculator
# ==========================================

@tool
def calculator(
    a: float,
    b: float,
    operation: str
) -> float:
    """
    Perform arithmetic calculations.

    operation:
    add
    subtract
    multiply
    divide
    """

    if operation == "add":

        return a + b

    elif operation == "subtract":

        return a - b

    elif operation == "multiply":

        return a * b

    elif operation == "divide":

        if b == 0:
            return "Cannot divide by zero"

        return a / b

    else:

        return "Unknown operation"


# ==========================================
# Tool collection
# ==========================================

tools = [
    search_tool,
    wikipedia_tool,
    calculator
]


# ==========================================
# LLM
# ==========================================

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
    cohere_api_key=cohere_api_key
)


# ==========================================
# Agent
# ==========================================

agent = create_agent(

    model=model,

    tools=tools,

    system_prompt="""

    You are a helpful AI assistant.

    You have access to several tools.

    Use Wikipedia for encyclopedia knowledge.

    Use DuckDuckGo when recent or
    current information is required.

    Use the calculator for arithmetic.

    Select tools automatically depending
    on the user's question.

    """
)


# ==========================================
# User question
# ==========================================

question = input(
    "\nAsk the AI Agent a question: "
)


# ==========================================
# Run Agent
# ==========================================

response = agent.invoke({

    "messages": [

        {
            "role": "user",
            "content": question
        }

    ]

})


# ==========================================
# Display final response
# ==========================================

print("\nAgent Response:\n")

print(
    response["messages"][-1].content
)