import os
from pathlib import Path

from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage


def load_env_file(env_path: str = ".env") -> None:
    """Load key/value pairs from a local .env file into os.environ."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
api_key = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
if not api_key:
    raise RuntimeError("Set CO_API_KEY or COHERE_API_KEY in your environment or .env file.")

# Initialize the vision model
llm = ChatCohere(model="command-a-vision-07-2025", cohere_api_key=api_key)

# Using an image URL
message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    ]
)
response = llm.invoke([message])
print("response 1")
print(response.content)

# Using a base64-encoded image with detail level

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this chart in detail"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KG...",
                "detail": "high"  # Options: "low", "high", or "auto" (default)
            }
        }
    ]
)
response = llm.invoke([message])
print("response 2")
print(response.content)

# Multiple images
message = HumanMessage(
    content=[
        {"type": "text", "text": "Compare these images"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
    ]
)
response = llm.invoke([message])
print("response 3")
print(response.content)
