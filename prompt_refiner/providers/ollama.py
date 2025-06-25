"""Ollama provider implementation"""

import json
from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderError


class OllamaProvider(BaseProvider):
    """Provider for Ollama local models"""

    def refine_prompt(
        self,
        prompt: str,
        focus_areas: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use Ollama to refine the prompt"""
        try:
            import httpx
        except ImportError as e:
            raise ProviderError("httpx package not installed. Run: pip install httpx") from e

        api_url = self.config.provider.ollama.get('api_url', 'http://localhost:11434')
        model = self.config.provider.ollama.get('model', 'llama3.2')
        temperature = self.config.provider.ollama.get('temperature', 0.7)
        retry_attempts = self.config.advanced.retry_attempts
        timeout = self.config.advanced.timeout_seconds

        refinement_prompt = f"""You are an expert at improving prompts for clarity and effectiveness.

Please refine the following prompt to make it clearer and more effective.

Original prompt: {prompt}

Focus areas: {', '.join(focus_areas) if focus_areas else 'clarity, specificity, actionability'}

Provide your response in JSON format with these fields:
- improved_prompt: The refined version of the prompt
- changes_made: Brief explanation of what was changed
- effectiveness_score: Rate the improvement (e.g., "8/10 - Much clearer")

Respond only with valid JSON."""

        for attempt in range(retry_attempts + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        f"{api_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": refinement_prompt,
                            "temperature": temperature,
                            "stream": False,
                            "format": "json"
                        }
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Parse the response
                    if 'response' in result:
                        return json.loads(result['response'])

                    return result

            except Exception as e:
                if attempt < retry_attempts:
                    continue
                raise ProviderError(f"Failed to connect to Ollama: {str(e)}") from e

    @staticmethod
    def is_available() -> bool:
        """Check if Ollama is running"""
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
