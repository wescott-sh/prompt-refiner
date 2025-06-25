"""Claude provider implementation"""

import subprocess
import json
import time
import shutil
from typing import Dict, Any

from .base import BaseProvider, ProviderRegistry


class ClaudeProvider(BaseProvider):
    """Provider for Claude CLI"""
    
    def refine(self, prompt: str, retry_attempts: int, timeout_seconds: int) -> Dict[str, str]:
        """Use Claude to refine the prompt"""
        for attempt in range(retry_attempts):
            try:
                result = subprocess.run(
                    ["claude", "--output-format", "json", "-p", prompt],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout_seconds
                )
                
                # Parse the JSON response
                response_data = json.loads(result.stdout)
                
                # Extract the actual result from the response structure
                if isinstance(response_data, dict) and "result" in response_data:
                    result_text = response_data["result"]
                    # Remove markdown code block if present
                    if result_text.startswith("```json"):
                        result_text = result_text[7:]
                    if result_text.endswith("```"):
                        result_text = result_text[:-3]
                    return json.loads(result_text.strip())
                
                return response_data
                
            except subprocess.TimeoutExpired:
                if attempt < retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError("Claude request timed out")
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise e
    
    def is_available(self) -> bool:
        """Check if Claude CLI is available"""
        return shutil.which('claude') is not None
    
    @property
    def name(self) -> str:
        """Return the name of this provider"""
        return 'claude'


# Register the provider
ProviderRegistry.register('claude', ClaudeProvider)