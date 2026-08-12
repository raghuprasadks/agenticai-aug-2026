import cohere

# Reuse the same setup as 1-cohere-chatapi.py
co = cohere.ClientV2(api_key="your cohere api key")


def get_answer_text(response) -> str:
    content_items = getattr(response.message, "content", []) or []
    text_parts = []

    for item in content_items:
        if getattr(item, "type", None) == "text" and hasattr(item, "text"):
            text_parts.append(item.text)

    if text_parts:
        return "\n".join(text_parts).strip()

    return "No text answer was returned by the model."


def main() -> None:
    model = "command-a-plus-05-2026"
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]

    print("Cohere Command Chat")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

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

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    }
                ],
            }
        )

        try:
            response = co.chat(
                messages=messages,
                temperature=0.6,
                model=model,
            )

            answer = get_answer_text(response)
            print(f"Assistant: {answer}\n")

            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                }
            )
        except Exception as error:
            print(f"Assistant error: {error}\n")


if __name__ == "__main__":
    main()
