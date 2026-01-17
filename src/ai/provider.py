from abc import ABC, abstractmethod
import requests
import json
from typing import Iterator

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

    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
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

    def generate_stream(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class OllamaProvider(AIProvider):
    def __init__(self):
        self.base_url = config.get("ollama.base_url")
        self.model = config.get("ollama.model")

    def generate(self, prompt: str) -> str:
        # Re-use stream for simplicity or keep separate
        full = ""
        for chunk in self.generate_stream(prompt):
            full += chunk
        return full

    def generate_stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True # Enable streaming
        }
        try:
            with requests.post(url, json=payload, stream=True) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.strip() == "data: [DONE]": 
                                break # OpenAI format done signal
                            
                            # Handle "data: " prefix if present (OpenAI style)
                            if decoded.startswith("data: "):
                                decoded = decoded[6:]
                                
                            try:
                                data = json.loads(decoded)
                                # OpenAI compatible format
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content: 
                                        yield content
                                # Native Ollama format
                                elif "message" in data:
                                    content = data["message"].get("content", "")
                                    if content: 
                                        yield content
                            except:
                                pass # Ignore parsing errors in stream
                else:
                    yield f"[Error: {resp.status_code}]"
        except Exception as e:
            yield f"[Connection Error: {str(e)}]"

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
