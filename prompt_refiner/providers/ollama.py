"""Ollama provider implementation"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any

from .base import BaseProvider, ProviderRegistry


class OllamaProvider(BaseProvider):
    """Provider for Ollama LLM"""
    
    @property
    def name(self) -> str:
        """Return the name of this provider"""
        return 'ollama'
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            url = self.config.get('api_url', 'http://localhost:11434')
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except:
            return False
    
    def refine(self, prompt: str, retry_attempts: int, timeout_seconds: int) -> Dict[str, str]:
        """Use Ollama to refine the prompt"""
        api_url = self.config.get('api_url', 'http://localhost:11434')
        model = self.config.get('model', 'llama2')
        temperature = self.config.get('temperature', 0.7)
        
        url = f"{api_url}/api/generate"
        
        for attempt in range(retry_attempts):
            try:
                data = json.dumps({
                    'model': model,
                    'prompt': prompt + "\n\nRespond with valid JSON only.",
                    'temperature': temperature,
                    'stream': False,
                    'format': 'json'
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Ollama error: {response.status}")
                    
                    result = json.loads(response.read().decode('utf-8'))
                    return json.loads(result['response'])
                
            except urllib.error.URLError as e:
                if attempt < retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError("Ollama request failed: " + str(e))
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise e


# Register the provider
ProviderRegistry.register('ollama', OllamaProvider)