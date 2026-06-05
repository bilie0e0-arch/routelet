from routelet.adapters.openai import OpenAIAdapter


class GroqAdapter(OpenAIAdapter):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url="https://api.groq.com/openai/v1")


class CerebrasAdapter(OpenAIAdapter):
    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, base_url="https://api.cerebras.ai/v1")


class OllamaAdapter(OpenAIAdapter):
    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        super().__init__(api_key="ollama", base_url=base_url)


class GoogleAdapter(OpenAIAdapter):
    """Uses Google AI Studio's OpenAI-compatible endpoint."""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
