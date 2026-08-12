"""
LangChain Demo — Core Concepts
================================
Concepts covered:
  1. LLM (Chat Model)       — Direct call to Cohere
  2. Prompt Templates       — Reusable, parameterised prompts
  3. Chains (LCEL)          — Pipe operator to link steps
  4. Conversation Memory    — Keep context across multiple turns
"""
#pip install langchain langchain-cohere python-dotenv -q
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatCohere(model="command-a-plus-05-2026")

# ─────────────────────────────────────────────────────────────
# 1. DIRECT LLM CALL
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DIRECT LLM CALL")
print("=" * 60)

response = llm.invoke("What is LangChain in one sentence?")
print(response.content)

# ─────────────────────────────────────────────────────────────
# 2. PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. PROMPT TEMPLATES")
print("=" * 60)

# A template that accepts variables — reusable across different inputs
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher. Explain concepts clearly and simply."),
    ("human",  "Explain {topic} in 2-3 sentences for a beginner."),
])

# Inspect what the formatted prompt looks like before sending it
formatted = prompt.format_messages(topic="large language models")
print("Formatted prompt (system):", formatted[0].content)
print("Formatted prompt (human) :", formatted[1].content)

response = llm.invoke(formatted)
print("\nLLM response:", response.content)

# ─────────────────────────────────────────────────────────────
# 3. CHAINS  (LangChain Expression Language — LCEL)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CHAINS  (prompt | llm)")
print("=" * 60)

# The | operator wires steps together: prompt → llm
# Output of one step automatically becomes input of the next.
chain = prompt | llm

result = chain.invoke({"topic": "vector databases"})
print(result.content)

# Chain with an output parser to extract plain text directly
from langchain_core.output_parsers import StrOutputParser

text_chain = prompt | llm | StrOutputParser()
plain_text = text_chain.invoke({"topic": "embeddings"})
print("\nWith StrOutputParser:", plain_text)

# ─────────────────────────────────────────────────────────────
# 4. CONVERSATION MEMORY  (multi-turn chat)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. CONVERSATION MEMORY")
print("=" * 60)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),  # ← injected message list
    ("human", "{question}"),
])

chat_chain = chat_prompt | llm | StrOutputParser()

# Maintain history manually — simplest and most transparent approach
history: list = []

def chat(question: str) -> str:
    answer = chat_chain.invoke({"history": history, "question": question})
    # Append the exchange so future turns have context
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer

print("User : What is Python?")
print("Bot  :", chat("What is Python?"))

print("\nUser : Can I use it for AI?")
print("Bot  :", chat("Can I use it for AI?"))  # references previous context

print("\nUser : Give me one library name for that.")
print("Bot  :", chat("Give me one library name for that."))  # still in context
