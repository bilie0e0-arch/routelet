JUDGE_MODEL = "gpt-4o-2024-08-06"

MODELS: dict[str, str] = {
    "strong_anthropic": "claude-sonnet-4-6-20260101",
    "cheap_anthropic": "claude-haiku-4-5-20251001",
    "strong_google": "gemini-2.0-flash",
    "cheap_groq": "llama-3.3-8b-instant",
    "mid_groq": "llama-3.3-70b-versatile",
    "local_tiny": "qwen2.5:1.5b",
}

FALLBACK_MODEL = MODELS["strong_anthropic"]

PROVIDER_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "ollama": "http://localhost:11434/v1",
}
