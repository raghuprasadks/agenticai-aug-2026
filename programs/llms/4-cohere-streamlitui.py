import json
from pathlib import Path

import cohere
import streamlit as st

# Reuse the same setup as previous programs
co = cohere.ClientV2(api_key="your cohere api key")

MEMORY_FILE = Path(__file__).with_name("cohere_memory.json")
MAX_HISTORY_MESSAGES = 30
MODEL = "command-a-plus-05-2026"


def get_answer_text(response) -> str:
    content_items = getattr(response.message, "content", []) or []
    text_parts = []

    for item in content_items:
        if getattr(item, "type", None) == "text" and hasattr(item, "text"):
            text_parts.append(item.text)

    if text_parts:
        return "\n".join(text_parts).strip()

    return "No text answer was returned by the model."


def load_memory() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    valid_messages = []
    for item in data:
        if isinstance(item, dict) and "role" in item and "content" in item:
            valid_messages.append(item)

    return valid_messages[-MAX_HISTORY_MESSAGES:]


def save_memory(messages: list[dict]) -> None:
    to_save = [message for message in messages if message.get("role") != "system"]
    to_save = to_save[-MAX_HISTORY_MESSAGES:]

    try:
        MEMORY_FILE.write_text(json.dumps(to_save, indent=2), encoding="utf-8")
    except OSError as error:
        st.warning(f"Could not save memory: {error}")


def build_messages(memory_messages: list[dict]) -> list[dict]:
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Use previous conversation when relevant.",
    }
    return [system_message, *memory_messages]


def to_ui_text(message: dict) -> str:
    content_items = message.get("content", [])
    if not isinstance(content_items, list):
        return ""

    text_parts = []
    for item in content_items:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))

    return "\n".join(part for part in text_parts if part).strip()


def init_state() -> None:
    if "memory_messages" not in st.session_state:
        st.session_state.memory_messages = load_memory()

    if "messages" not in st.session_state:
        st.session_state.messages = build_messages(st.session_state.memory_messages)


def reset_memory() -> None:
    st.session_state.memory_messages = []
    st.session_state.messages = build_messages([])
    save_memory([])


def ask_model(user_text: str) -> str:
    user_message = {
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    }
    st.session_state.messages.append(user_message)

    response = co.chat(
        messages=st.session_state.messages,
        temperature=0.6,
        model=MODEL,
    )

    answer = get_answer_text(response)

    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }
    st.session_state.messages.append(assistant_message)

    st.session_state.memory_messages = [
        message for message in st.session_state.messages if message.get("role") != "system"
    ][-MAX_HISTORY_MESSAGES:]

    save_memory(st.session_state.memory_messages)
    st.session_state.messages = build_messages(st.session_state.memory_messages)

    return answer


def main() -> None:
    st.set_page_config(page_title="Cohere Memory Chat", page_icon="💬", layout="centered")
    st.title("Cohere Chat With Memory")
    st.caption("Built from program 3 with Streamlit UI")

    init_state()

    with st.sidebar:
        st.subheader("Memory")
        st.write(f"Saved messages: {len(st.session_state.memory_messages)}")
        if st.button("Reset Memory", use_container_width=True):
            reset_memory()
            st.success("Memory cleared")

    for message in st.session_state.memory_messages:
        role = message.get("role", "assistant")
        text = to_ui_text(message)
        if role in {"user", "assistant"} and text:
            with st.chat_message(role):
                st.markdown(text)

    prompt = st.chat_input("Ask something...")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_model(prompt)
                    st.markdown(answer)
                except Exception as error:
                    st.error(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
