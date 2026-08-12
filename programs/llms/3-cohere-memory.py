import json
from pathlib import Path

import cohere

# Reuse the same setup as previous programs
co = cohere.ClientV2(api_key="your cohere api key")

MEMORY_FILE = Path(__file__).with_name("cohere_memory.json")
MAX_HISTORY_MESSAGES = 30


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
        print(f"Warning: could not save memory: {error}")


def build_messages(memory_messages: list[dict]) -> list[dict]:
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Use previous conversation when relevant.",
    }
    return [system_message, *memory_messages]


def main() -> None:
    model = "command-a-plus-05-2026"
    memory_messages = load_memory()
    messages = build_messages(memory_messages)

    print("Cohere Command Chat + Memory")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("Type '/reset' to clear saved memory.\n")

    if memory_messages:
        print(f"Loaded memory with {len(memory_messages)} past messages.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_text.lower() == "/reset":
            memory_messages = []
            messages = build_messages(memory_messages)
            save_memory(memory_messages)
            print("Memory cleared.\n")
            continue

        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
                }
            ],
        }
        messages.append(user_message)

        try:
            response = co.chat(
                messages=messages,
                temperature=0.6,
                model=model,
            )

            answer = get_answer_text(response)
            print(f"Assistant: {answer}\n")

            assistant_message = {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
            }
            messages.append(assistant_message)

            memory_messages = [m for m in messages if m.get("role") != "system"]
            save_memory(memory_messages)

            # Keep in-memory context bounded for long sessions.
            messages = build_messages(memory_messages[-MAX_HISTORY_MESSAGES:])
        except Exception as error:
            print(f"Assistant error: {error}\n")


if __name__ == "__main__":
    main()
