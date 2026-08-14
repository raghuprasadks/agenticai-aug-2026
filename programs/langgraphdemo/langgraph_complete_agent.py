# ============================================================
# COMPLETE LANGGRAPH AGENTIC AI PROGRAM
#
# Components:
# 1. Cohere LLM
# 2. PDF RAG + Chroma
# 3. Planner
# 4. Decision Router
# 5. Web Search
# 6. Calculator
# 7. Answer Generator
# 8. Reviewer / Critic
# 9. Revision Loop
# 10. Human Approval
# 11. LangGraph Memory
# ============================================================


import os
import re
import json
from typing import TypedDict

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# LANGCHAIN / LANGGRAPH IMPORTS
# ============================================================

from langchain_cohere import (
    ChatCohere,
    CohereEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_community.tools import (
    DuckDuckGoSearchRun
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    InMemorySaver
)

from langgraph.types import (
    interrupt,
    Command
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "documents/company_policy.pdf"

CHROMA_DIRECTORY = "./chroma_db"

COLLECTION_NAME = "company_documents"

THREAD_ID = "classroom-demo-001"


# ============================================================
# CHECK COHERE API KEY
# ============================================================

if not os.getenv("COHERE_API_KEY"):

    print("\nERROR:")
    print("COHERE_API_KEY is not configured.")

    print("\nWindows Command Prompt:")
    print(
        "set COHERE_API_KEY=your_cohere_api_key"
    )

    print("\nPowerShell:")
    print(
        '$env:COHERE_API_KEY="your_cohere_api_key"'
    )

    exit()


# ============================================================
# CREATE COHERE LLM
# ============================================================

llm = ChatCohere(

    model="command-a-03-2025",

    temperature=0
)


# ============================================================
# CREATE COHERE EMBEDDINGS
# ============================================================

embeddings = CohereEmbeddings(

    model="embed-v4.0"
)


# ============================================================
# CREATE / LOAD VECTOR DATABASE
# ============================================================

def initialize_vector_database():

    """
    Create Chroma database if required.

    If a PDF exists, load and embed it.
    Otherwise connect to an existing database.
    """

    print(
        "\nInitializing knowledge base..."
    )


    if os.path.exists(PDF_FILE):

        print(
            f"Loading PDF: {PDF_FILE}"
        )


        loader = PyPDFLoader(
            PDF_FILE
        )


        documents = loader.load()


        print(
            f"Pages loaded: {len(documents)}"
        )


        text_splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=800,

                chunk_overlap=150
            )
        )


        chunks = (
            text_splitter
            .split_documents(
                documents
            )
        )


        print(
            f"Chunks created: {len(chunks)}"
        )


        vector_store = (
            Chroma.from_documents(

                documents=chunks,

                embedding=embeddings,

                collection_name=
                COLLECTION_NAME,

                persist_directory=
                CHROMA_DIRECTORY
            )
        )


        print(
            "Vector database ready."
        )


    else:

        print(
            "PDF not found."
        )

        print(
            "Connecting to existing "
            "Chroma database..."
        )


        vector_store = Chroma(

            collection_name=
            COLLECTION_NAME,

            embedding_function=
            embeddings,

            persist_directory=
            CHROMA_DIRECTORY
        )


    return vector_store


# ============================================================
# INITIALIZE VECTOR DATABASE
# ============================================================

vector_store = (
    initialize_vector_database()
)


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = (
    vector_store.as_retriever(

        search_kwargs={

            "k": 4
        }
    )
)


# ============================================================
# WEB SEARCH TOOL
# ============================================================

web_search = DuckDuckGoSearchRun()


# ============================================================
# DEFINE LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):

    question: str

    plan: str

    next_action: str

    retrieved_context: str

    tool_result: str

    draft_answer: str

    review: str

    revision_count: int

    approved: bool

    final_answer: str


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_text(response):

    """
    Safely obtain text from the
    model response.
    """

    if isinstance(
        response.content,
        str
    ):

        return response.content


    return str(
        response.content
    )


# ============================================================
# NODE 1
# PLANNER
# ============================================================

def planner_node(
    state: AgentState
):

    print(
        "\n[PLANNER]"
    )


    prompt = f"""

You are a planning agent.

User question:

{state['question']}


Create a short plan for answering
the question.

Possible capabilities available:

1. Internal document / RAG search
2. Current web search
3. Mathematical calculation
4. Direct LLM reasoning


Return a concise step-by-step plan.

"""


    response = llm.invoke(
        prompt
    )


    plan = get_text(
        response
    )


    print(
        "\nPlan:"
    )

    print(
        plan
    )


    return {

        "plan":
            plan,

        "revision_count":
            state.get(
                "revision_count",
                0
            )
    }


# ============================================================
# NODE 2
# DECISION
# ============================================================

def decision_node(
    state: AgentState
):

    print(
        "\n[DECISION]"
    )


    prompt = f"""

You are a routing agent.

User question:

{state['question']}


Plan:

{state['plan']}


Choose the best next action.

Available actions:

rag
web
calculator
direct


Rules:

rag:
Use for internal company documents,
policies, manuals or private
knowledge.

web:
Use when recent, latest, current
or external Internet information
is required.

calculator:
Use for mathematical calculations.

direct:
Use when the language model can
answer without another tool.


Return ONLY one word:

rag
web
calculator
direct

"""


    response = llm.invoke(
        prompt
    )


    decision = (
        get_text(response)
        .strip()
        .lower()
    )


    # Defensive cleanup

    if "calculator" in decision:

        decision = "calculator"

    elif "rag" in decision:

        decision = "rag"

    elif "web" in decision:

        decision = "web"

    else:

        decision = "direct"


    print(
        f"Selected action: {decision}"
    )


    return {

        "next_action":
            decision
    }


# ============================================================
# ROUTER
# ============================================================

def action_router(
    state: AgentState
):

    return (
        state["next_action"]
    )


# ============================================================
# NODE 3
# RAG
# ============================================================

def rag_node(
    state: AgentState
):

    print(
        "\n[RAG SEARCH]"
    )


    documents = (
        retriever.invoke(
            state["question"]
        )
    )


    if not documents:

        context = (
            "No relevant information "
            "was found in the "
            "internal documents."
        )


    else:

        parts = []


        for i, doc in enumerate(
            documents,
            start=1
        ):

            source = (
                doc.metadata
                .get(
                    "source",
                    "Unknown"
                )
            )


            page = (
                doc.metadata
                .get(
                    "page",
                    "Unknown"
                )
            )


            text = f"""

DOCUMENT {i}

Source:
{source}

Page:
{page}

Content:
{doc.page_content}

"""


            parts.append(
                text
            )


        context = (
            "\n".join(parts)
        )


    print(
        "Retrieved document context."
    )


    return {

        "retrieved_context":
            context,

        "tool_result":
            context
    }


# ============================================================
# NODE 4
# WEB SEARCH
# ============================================================

def web_node(
    state: AgentState
):

    print(
        "\n[WEB SEARCH]"
    )


    try:

        result = (
            web_search.invoke(
                state["question"]
            )
        )


    except Exception as e:

        result = (
            "Web search failed: "
            + str(e)
        )


    print(
        "Web search completed."
    )


    return {

        "tool_result":
            result
    }


# ============================================================
# NODE 5
# CALCULATOR
# ============================================================

def calculator_node(
    state: AgentState
):

    print(
        "\n[CALCULATOR]"
    )


    prompt = f"""

Extract only the mathematical
expression required to answer
the following question.

Question:

{state['question']}


Examples:

Question:
What is 25 multiplied by 12?

Output:

25 * 12


Question:
What is 100 divided by 5?

Output:

100 / 5


Return ONLY the arithmetic
expression.

"""


    response = llm.invoke(
        prompt
    )


    expression = (
        get_text(response)
        .strip()
    )


    print(
        f"Expression: {expression}"
    )


    # Allow only basic arithmetic

    pattern = (
        r"[0-9+\-*/(). %]+"
    )


    if not re.fullmatch(
        pattern,
        expression
    ):

        result = (
            "The generated arithmetic "
            "expression was rejected "
            "for safety."
        )


    else:

        try:

            result = str(

                eval(

                    expression,

                    {
                        "__builtins__":
                            {}
                    },

                    {}
                )
            )


        except Exception as e:

            result = (
                "Calculation failed: "
                + str(e)
            )


    print(
        f"Result: {result}"
    )


    return {

        "tool_result":
            result
    }


# ============================================================
# NODE 6
# DIRECT LLM REASONING
# ============================================================

def direct_node(
    state: AgentState
):

    print(
        "\n[DIRECT REASONING]"
    )


    prompt = f"""

Answer the following question
clearly and accurately.

Question:

{state['question']}

"""


    response = llm.invoke(
        prompt
    )


    result = get_text(
        response
    )


    return {

        "tool_result":
            result
    }


# ============================================================
# NODE 7
# ANSWER SYNTHESIS
# ============================================================

def agent_node(
    state: AgentState
):

    print(
        "\n[ANSWER GENERATOR]"
    )


    prompt = f"""

You are the answer generation
agent.

User Question:

{state['question']}


Plan:

{state['plan']}


Evidence / Tool Result:

{state['tool_result']}


Create a clear and concise
answer for the user.


Important:

Do not invent facts.

If the evidence does not contain
enough information, clearly say so.

"""


    response = llm.invoke(
        prompt
    )


    answer = get_text(
        response
    )


    print(
        "\nDraft Answer:"
    )

    print(
        answer
    )


    return {

        "draft_answer":
            answer
    }


# ============================================================
# NODE 8
# REVIEWER / CRITIC
# ============================================================

def reviewer_node(
    state: AgentState
):

    print(
        "\n[REVIEWER]"
    )


    prompt = f"""

You are an independent reviewer.

Evaluate the draft answer.


USER QUESTION:

{state['question']}


DRAFT ANSWER:

{state['draft_answer']}


AVAILABLE EVIDENCE:

{state['tool_result']}


Check:

1. Correctness
2. Relevance
3. Completeness
4. Unsupported claims
5. Whether the answer actually
   answers the question


If the answer is acceptable,
return exactly:

APPROVED


Otherwise return:

REVISE: <short explanation>

"""


    response = llm.invoke(
        prompt
    )


    review = (
        get_text(response)
        .strip()
    )


    print(
        f"Review: {review}"
    )


    return {

        "review":
            review
    }


# ============================================================
# REVIEW ROUTER
# ============================================================

def review_router(
    state: AgentState
):

    # Prevent an infinite
    # reviewer/revision loop

    revision_count = (
        state.get(
            "revision_count",
            0
        )
    )


    if (
        state["review"]
        .upper()
        .startswith(
            "APPROVED"
        )
    ):

        return "human_approval"


    if revision_count >= 2:

        print(
            "\nMaximum revision "
            "limit reached."
        )

        return "human_approval"


    return "revise"


# ============================================================
# NODE 9
# REVISION
# ============================================================

def revise_node(
    state: AgentState
):

    print(
        "\n[REVISION]"
    )


    count = (
        state.get(
            "revision_count",
            0
        )
        + 1
    )


    prompt = f"""

Improve the following answer
using the reviewer's feedback.


QUESTION:

{state['question']}


CURRENT ANSWER:

{state['draft_answer']}


REVIEWER FEEDBACK:

{state['review']}


AVAILABLE EVIDENCE:

{state['tool_result']}


Produce only the improved answer.

"""


    response = llm.invoke(
        prompt
    )


    revised_answer = (
        get_text(response)
    )


    print(
        f"Revision #{count}"
    )


    print(
        revised_answer
    )


    return {

        "draft_answer":
            revised_answer,

        "revision_count":
            count
    }


# ============================================================
# NODE 10
# HUMAN APPROVAL
# ============================================================

def human_approval_node(
    state: AgentState
):

    print(
        "\n[HUMAN APPROVAL REQUIRED]"
    )


    human_decision = interrupt({

        "question":
            state["question"],

        "draft_answer":
            state["draft_answer"],

        "review":
            state["review"],

        "instruction":
            "Type approve or reject."
    })


    approved = (

        str(human_decision)
        .strip()
        .lower()
        == "approve"
    )


    return {

        "approved":
            approved
    }


# ============================================================
# HUMAN APPROVAL ROUTER
# ============================================================

def approval_router(
    state: AgentState
):

    if state["approved"]:

        return "final"


    return "revise"


# ============================================================
# NODE 11
# FINAL ANSWER
# ============================================================

def final_node(
    state: AgentState
):

    print(
        "\n[FINAL ANSWER]"
    )


    return {

        "final_answer":
            state[
                "draft_answer"
            ]
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


# ============================================================
# ADD NODES
# ============================================================

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "decision",
    decision_node
)

builder.add_node(
    "rag",
    rag_node
)

builder.add_node(
    "web",
    web_node
)

builder.add_node(
    "calculator",
    calculator_node
)

builder.add_node(
    "direct",
    direct_node
)

builder.add_node(
    "agent",
    agent_node
)

builder.add_node(
    "reviewer",
    reviewer_node
)

builder.add_node(
    "revise",
    revise_node
)

builder.add_node(
    "human_approval",
    human_approval_node
)

builder.add_node(
    "final",
    final_node
)


# ============================================================
# CONNECT GRAPH
# ============================================================

builder.add_edge(
    START,
    "planner"
)


builder.add_edge(
    "planner",
    "decision"
)


# ============================================================
# CONDITIONAL TOOL ROUTING
# ============================================================

builder.add_conditional_edges(

    "decision",

    action_router,

    {

        "rag":
            "rag",

        "web":
            "web",

        "calculator":
            "calculator",

        "direct":
            "direct"
    }
)


# ============================================================
# TOOL RESULTS GO TO ANSWER GENERATOR
# ============================================================

builder.add_edge(
    "rag",
    "agent"
)

builder.add_edge(
    "web",
    "agent"
)

builder.add_edge(
    "calculator",
    "agent"
)

builder.add_edge(
    "direct",
    "agent"
)


# ============================================================
# ANSWER -> REVIEWER
# ============================================================

builder.add_edge(
    "agent",
    "reviewer"
)


# ============================================================
# REVIEW ROUTING
# ============================================================

builder.add_conditional_edges(

    "reviewer",

    review_router,

    {

        "human_approval":
            "human_approval",

        "revise":
            "revise"
    }
)


# ============================================================
# REVISED ANSWER GOES BACK TO REVIEWER
# ============================================================

builder.add_edge(
    "revise",
    "reviewer"
)


# ============================================================
# HUMAN APPROVAL ROUTING
# ============================================================

builder.add_conditional_edges(

    "human_approval",

    approval_router,

    {

        "final":
            "final",

        "revise":
            "revise"
    }
)


# ============================================================
# FINISH
# ============================================================

builder.add_edge(
    "final",
    END
)


# ============================================================
# MEMORY / CHECKPOINTER
# ============================================================

memory = InMemorySaver()


# ============================================================
# COMPILE GRAPH
# ============================================================

graph = builder.compile(

    checkpointer=memory
)


# ============================================================
# THREAD CONFIGURATION
# ============================================================

config = {

    "configurable": {

        "thread_id":
            THREAD_ID
    }
}


# ============================================================
# FUNCTION TO RUN AGENT
# ============================================================

def run_agent(question):

    initial_state = {

        "question":
            question,

        "plan":
            "",

        "next_action":
            "",

        "retrieved_context":
            "",

        "tool_result":
            "",

        "draft_answer":
            "",

        "review":
            "",

        "revision_count":
            0,

        "approved":
            False,

        "final_answer":
            ""
    }


    # --------------------------------------------------------
    # START GRAPH
    # --------------------------------------------------------

    result = graph.invoke(

        initial_state,

        config=config
    )


    # --------------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------------

    while "__interrupt__" in result:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "HUMAN REVIEW"
        )

        print(
            "=" * 60
        )


        # Show latest saved state

        snapshot = (
            graph.get_state(
                config
            )
        )


        current_state = (
            snapshot.values
        )


        print(
            "\nQUESTION:\n"
        )

        print(
            current_state
            .get(
                "question",
                ""
            )
        )


        print(
            "\nPROPOSED ANSWER:\n"
        )

        print(
            current_state
            .get(
                "draft_answer",
                ""
            )
        )


        print(
            "\nREVIEWER COMMENT:\n"
        )

        print(
            current_state
            .get(
                "review",
                ""
            )
        )


        decision = input(

            "\nApprove answer? "
            "(approve/reject): "

        ).strip().lower()


        if decision not in [
            "approve",
            "reject"
        ]:

            decision = "reject"


        # Resume graph

        result = graph.invoke(

            Command(
                resume=decision
            ),

            config=config
        )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    final_answer = (
        result.get(
            "final_answer",
            ""
        )
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 60
    )


    print(
        final_answer
    )


    return final_answer


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 65
    )

    print(
        "LANGGRAPH AGENTIC AI DEMO"
    )

    print(
        "Cohere + RAG + Web + "
        "Calculator + Reviewer + "
        "Human Approval"
    )

    print(
        "=" * 65
    )


    print(
        "\nType 'exit' to stop."
    )


    while True:

        question = input(
            "\nYou: "
        )


        if question.lower() in [

            "exit",
            "quit"

        ]:

            print(
                "\nAgent stopped."
            )

            break


        run_agent(
            question
        )