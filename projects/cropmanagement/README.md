# Crop Management RAG Chatbot

A simple farmer-focused chatbot built with Python, Streamlit, Cohere, and ChromaDB. It lets farmers upload crop-management documents, index them into a vector database, and ask natural-language questions about crop care, irrigation, disease management, soil health, pest control, and agronomy guidance.

## Features

- Multi-file document upload
- Supports PDF, DOCX, TXT, and MD files
- Local vector storage with ChromaDB
- Semantic search using Cohere embeddings
- LLM-based answers using Cohere chat model
- Farmer-friendly interface in Streamlit

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   myenv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Add your Cohere API key:
   ```bash
   copy .env.example .env
   ```
   Then update `.env` with your real key.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The app stores indexed crop knowledge in a local ChromaDB folder named `chroma_db`.
- You can upload multiple documents before asking questions.
- If the app cannot find a Cohere API key, it will show a clear warning and stop until the key is configured.
