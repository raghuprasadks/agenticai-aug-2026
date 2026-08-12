"""
Agents Demo  (LangChain 1.x)
=============================
Concepts covered:
  1. Tools        — Functions the LLM can choose to call
  2. create_agent — LangGraph-backed agent loop (Reason → Act → Observe)
  3. Streaming    — See each step as the agent produces it
"""

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain.agents import create_agent   # new API in LangChain 1.x
from langchain_core.tools import tool
import math
import datetime

load_dotenv()

llm = ChatCohere(model="command-a-plus-05-2026")

# ─────────────────────────────────────────────────────────────
# 1. DEFINE TOOLS  (functions the agent can choose to call)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. TOOLS")
print("=" * 60)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this for any arithmetic or math question.
    Example expressions: '2 + 2', '15 * 8', 'sqrt(144)', '2 ** 10'
    """
    # Restrict to safe math functions only
    safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_globals["__builtins__"] = {}
    try:
        result = eval(expression, safe_globals)  # noqa: S307 — safe_globals restricts scope
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def get_current_date(timezone: str = "UTC") -> str:
    """Return today's date and current time.
    Use this when the user asks about the current date, day, or time.
    """
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%A, %d %B %Y, %H:%M:%S')} (local)"

@tool
def word_counter(text: str) -> str:
    """Count the number of words, characters, and sentences in a text.
    Use this when the user asks to analyse or count things in a piece of text.
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.
    Supported conversions:
      - km  ↔ miles
      - kg  ↔ pounds
      - celsius ↔ fahrenheit
    """
    conversions = {
        ("km",      "miles"):      lambda v: v * 0.621371,
        ("miles",   "km"):         lambda v: v * 1.60934,
        ("kg",      "pounds"):     lambda v: v * 2.20462,
        ("pounds",  "kg"):         lambda v: v / 2.20462,
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Conversion from {from_unit} to {to_unit} is not supported."
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"

tools = [calculator, get_current_date, word_counter, unit_converter]

print("Available tools:")
for t in tools:
    print(f"  - {t.name}: {t.description.splitlines()[0]}")

# ─────────────────────────────────────────────────────────────
# 2. CREATE THE AGENT  (LangChain 1.x API)
#    create_agent returns a LangGraph CompiledStateGraph that
#    automatically loops: call LLM → call tools → repeat until done.
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CREATING THE AGENT")
print("=" * 60)

llm = ChatCohere(model="command-a-plus-05-2026")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the tools whenever they are relevant.",
)

print("Agent created with tool-calling capability.")

# ─────────────────────────────────────────────────────────────
# 3. RUN THE AGENT  (watch it reason and pick tools)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. AGENT IN ACTION")
print("=" * 60)

queries = [
    "What is 15% of 8400?",
    "What day is today?",
    "Convert 100 km to miles and also convert 37 celsius to fahrenheit.",
    "Count the words in this sentence: The quick brown fox jumps over the lazy dog.",
]

for query in queries:
    print(f"\n{'─' * 60}")
    print(f"User: {query}")
    print('─' * 60)

    # stream() yields each step: tool calls, tool results, and final answer
    for step in agent.stream({"messages": [{"role": "user", "content": query}]}):
        # Each step is a dict with the node name as key
        for node, output in step.items():
            msgs = output.get("messages", [])
            for msg in msgs:
                msg_type = type(msg).__name__
                if msg_type == "AIMessage":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [Tool call] {tc['name']}({tc['args']})")
                    elif msg.content:
                        print(f"  Final Answer: {msg.content}")
                elif msg_type == "ToolMessage":
                    print(f"  [Tool result] {msg.content}")
