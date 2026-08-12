import json
from pathlib import Path

import cohere
import pyttsx3
import speech_recognition as sr

# Reuse the same setup as previous programs
co = cohere.ClientV2(api_key="your cohere api key")

MEMORY_FILE = Path(__file__).with_name("cohere_memory.json")
MAX_HISTORY_MESSAGES = 30
MODEL = "command-a-plus-05-2026"


class VoiceBot:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 175)
        self.tts.setProperty("volume", 1.0)

        memory_messages = load_memory()
        self.messages = build_messages(memory_messages)

    def speak(self, text: str) -> None:
        print(f"Assistant: {text}\n")
        self.tts.say(text)
        self.tts.runAndWait()

    def listen(self) -> str | None:
        with sr.Microphone() as source:
            print("Listening... (speak now)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=20)
            except sr.WaitTimeoutError:
                print("No speech detected. Try again.\n")
                return None

        try:
            text = self.recognizer.recognize_google(audio)
            text = text.strip()
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand your voice. Try again.\n")
            return None
        except sr.RequestError as error:
            print(f"Speech recognition service error: {error}\n")
            return None

    def reset_memory(self) -> None:
        self.messages = build_messages([])
        save_memory([])
        self.speak("Memory cleared.")

    def ask_model(self, user_text: str) -> str:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
        self.messages.append(user_message)

        response = co.chat(
            messages=self.messages,
            temperature=0.6,
            model=MODEL,
        )

        answer = get_answer_text(response)

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
        self.messages.append(assistant_message)

        memory_messages = [m for m in self.messages if m.get("role") != "system"]
        memory_messages = memory_messages[-MAX_HISTORY_MESSAGES:]
        save_memory(memory_messages)

        self.messages = build_messages(memory_messages)
        return answer


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
    print("Cohere Voice Chat + Memory")
    print("Speak your question when prompted.")
    print("Say 'exit' or 'quit' to stop.")
    print("Say 'reset memory' to clear saved context.\n")

    bot = VoiceBot()

    if len(bot.messages) > 1:
        print(f"Loaded memory with {len(bot.messages) - 1} past messages.\n")

    bot.speak("Voice chat is ready. Ask your question.")

    while True:
        user_text = bot.listen()
        if not user_text:
            continue

        command = user_text.lower()
        if command in {"exit", "quit", "stop"}:
            bot.speak("Goodbye.")
            break

        if command in {"reset memory", "clear memory"}:
            bot.reset_memory()
            continue

        try:
            answer = bot.ask_model(user_text)
            bot.speak(answer)
        except Exception as error:
            message = f"Assistant error: {error}"
            print(f"{message}\n")
            bot.speak("Sorry, I had an error while getting the answer.")


if __name__ == "__main__":
    main()
