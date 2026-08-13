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

llm = ChatCohere(cohere_api_key=api_key)

messages = [HumanMessage(content="Hello, can you introduce yourself?")]
print(llm.invoke(messages))