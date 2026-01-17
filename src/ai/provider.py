from abc import ABC, abstractmethod
import requests
# Try importing openai, handle if missing
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..core.config_loader import config

class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(AIProvider):
    def __init__(self):
        if not OpenAI:
            raise ImportError("openai package not installed.")
        self.client = OpenAI(
            api_key=config.get("openai.api_key"),
            base_url=config.get("openai.base_url")
        )
        self.model = config.get("openai.model")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""

class OllamaProvider(AIProvider):
    def __init__(self):
        self.base_url = config.get("ollama.base_url")
        self.model = config.get("ollama.model")

    def generate(self, prompt: str) -> str:
        # Simple REST API for Ollama (OpenAI Compatible Endpoint)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                # Try OpenAI format first
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "")
                # Fallback to Ollama native format just in case
                if "message" in data:
                    return data.get("message", {}).get("content", "")
                return str(data) # Debug if unknown format
            else:
                return f"Error from Ollama: {resp.text}"
        except Exception as e:
            return f"Ollama Connection Error: {str(e)}"

class AIProviderFactory:
    @staticmethod
    def get_provider() -> AIProvider:
        provider_type = config.get("ai_provider", "ollama")
        
        if provider_type == "openai":
            return OpenAIProvider()
        elif provider_type == "ollama":
            return OllamaProvider()
        else:
            raise ValueError(f"Unknown AI provider: {provider_type}")
