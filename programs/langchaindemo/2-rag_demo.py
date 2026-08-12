"""
RAG Demo — Retrieval-Augmented Generation
==========================================
Concepts covered:
  1. Document Loading       — Read and prepare documents
  2. Text Splitting         — Break docs into manageable chunks
  3. Embeddings             — Convert text to vectors
  4. Vector Store           — Store & retrieve similar docs
  5. RAG Chain              — Combine retrieval + generation
"""
#pip install langchain-community chromadb -q
#pip install langchain_chroma


from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1. SAMPLE DOCUMENTS (pretend we loaded these from files)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DOCUMENTS")
print("=" * 60)

documents = [
    Document(
        page_content="""Python is a high-level, interpreted programming language known for 
        its simplicity and readability. It supports multiple programming paradigms including 
        procedural, object-oriented, and functional programming. Python has a vast ecosystem 
        of libraries for data science, web development, and automation.""",
        metadata={"source": "python_guide.txt", "topic": "Python Basics"}
    ),
    Document(
        page_content="""Machine Learning is a subset of Artificial Intelligence that enables 
        systems to learn and improve from experience without being explicitly programmed. 
        It uses algorithms and statistical models to identify patterns in data. Common applications 
        include image recognition, natural language processing, and recommendation systems.""",
        metadata={"source": "ml_guide.txt", "topic": "Machine Learning"}
    ),
    Document(
        page_content="""LangChain is a framework for developing applications powered by language models. 
        It enables easy integration of language models with other tools and data sources. LangChain 
        provides abstractions and components for building chains, agents, and retrieval-augmented 
        generation systems. It supports multiple LLM providers like OpenAI, Cohere, and Hugging Face.""",
        metadata={"source": "langchain_guide.txt", "topic": "LangChain"}
    ),
    Document(
        page_content="""Deep Learning is a branch of Machine Learning based on artificial neural networks 
        with multiple layers. It has revolutionized computer vision, natural language processing, and 
        speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras. Deep learning 
        requires large amounts of data and computational power, typically GPUs or TPUs.""",
        metadata={"source": "deeplearning_guide.txt", "topic": "Deep Learning"}
    ),
]

print(f"Loaded {len(documents)} documents")
for doc in documents:
    print(f"  - {doc.metadata['topic']}: {doc.page_content[:50]}...")

# ─────────────────────────────────────────────────────────────
# 2. TEXT SPLITTING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. TEXT SPLITTING")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # max characters per chunk
    chunk_overlap=50,    # overlap to maintain context across chunks
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
print(f"Sample chunk: {chunks[0].page_content[:100]}...")

# ─────────────────────────────────────────────────────────────
# 3. EMBEDDINGS & VECTOR STORE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. EMBEDDINGS & VECTOR STORE (CHROMA)")
print("=" * 60)

embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Create Chroma vector store from chunks
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_docs"
)

print(f"Vector store created with {len(chunks)} embeddings")

# ─────────────────────────────────────────────────────────────
# 4. RETRIEVAL (Query → Find Similar Docs)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. RETRIEVAL")
print("=" * 60)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # retrieve top 2

query = "What is LangChain?"
retrieved = retriever.invoke(query)

print(f"Query: '{query}'")
print(f"Retrieved {len(retrieved)} relevant documents:")
for i, doc in enumerate(retrieved, 1):
    print(f"\n  [{i}] Topic: {doc.metadata['topic']}")
    print(f"      {doc.page_content[:80]}...")

# ─────────────────────────────────────────────────────────────
# 5. RAG CHAIN (Retrieval + Generation)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RAG CHAIN (Retrieval → LLM Generation)")
print("=" * 60)

llm = ChatCohere(model="command-a-plus-05-2026")

# Create a prompt that uses the retrieved context
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer:""")

# Function to format retrieved docs as context
def format_docs(docs):
    return "\n\n".join([f"[{doc.metadata['topic']}]\n{doc.page_content}" for doc in docs])

# Build the RAG chain: query → retriever → format → prompt → llm → parser
rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Test RAG on different questions
test_questions = [
    "What is LangChain?",
    "Tell me about Machine Learning",
    "What programming languages are mentioned?"
]

for q in test_questions:
    print(f"\nUser: {q}")
    answer = rag_chain.invoke(q)
    print(f"Bot : {answer}\n")
