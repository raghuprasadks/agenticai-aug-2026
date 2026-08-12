from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import cohere
import pyttsx3
import requests
import speech_recognition as sr

# Reuse the same setup style as previous programs
co = cohere.ClientV2(api_key="your cohere api key")
MODEL = "command-a-plus-05-2026"

OUTPUT_DIR = Path(__file__).with_name("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_answer_text(response) -> str:
    content_items = getattr(response.message, "content", []) or []
    text_parts = []

    for item in content_items:
        if getattr(item, "type", None) == "text" and hasattr(item, "text"):
            text_parts.append(item.text)

    if text_parts:
        return "\n".join(text_parts).strip()

    return ""


class VoiceImageBot:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 175)
        self.tts.setProperty("volume", 1.0)

    def speak(self, text: str) -> None:
        print(f"Assistant: {text}\n")
        self.tts.say(text)
        self.tts.runAndWait()

    def listen(self) -> str | None:
        with sr.Microphone() as source:
            print("Listening... speak your image idea.")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=6, phrase_time_limit=20)
            except sr.WaitTimeoutError:
                print("No speech detected. Try again.\n")
                return None

        try:
            text = self.recognizer.recognize_google(audio).strip()
            print(f"You: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand your voice. Try again.\n")
            return None
        except sr.RequestError as error:
            print(f"Speech recognition service error: {error}\n")
            return None

    def refine_prompt_with_cohere(self, voice_text: str) -> str:
        response = co.chat(
            model=MODEL,
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite user requests into one concise, vivid image generation prompt. Return only the prompt text.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": voice_text}],
                },
            ],
        )

        prompt = get_answer_text(response)
        if not prompt:
            return voice_text

        return prompt

    def generate_image(self, prompt: str) -> Path:
        encoded_prompt = quote_plus(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux&nologo=true"

        result = requests.get(image_url, timeout=60)
        result.raise_for_status()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = OUTPUT_DIR / f"cohere_voice_image_{timestamp}.png"
        file_path.write_bytes(result.content)
        return file_path


def main() -> None:
    print("Cohere Voice To Image")
    print("Speak what image you want.")
    print("Say 'exit' or 'quit' to stop.\n")

    bot = VoiceImageBot()
    bot.speak("Voice image bot is ready. Tell me what image you want.")

    while True:
        user_text = bot.listen()
        if not user_text:
            continue

        command = user_text.lower()
        if command in {"exit", "quit", "stop"}:
            bot.speak("Goodbye.")
            break

        try:
            bot.speak("Creating your image prompt.")
            prompt = bot.refine_prompt_with_cohere(user_text)
            print(f"Image prompt: {prompt}\n")

            bot.speak("Generating image now.")
            image_path = bot.generate_image(prompt)

            bot.speak(f"Image generated successfully. Saved as {image_path.name}")
            print(f"Saved image: {image_path}\n")
        except Exception as error:
            print(f"Error: {error}\n")
            bot.speak("Sorry, I could not generate the image. Please try again.")


if __name__ == "__main__":
    main()
