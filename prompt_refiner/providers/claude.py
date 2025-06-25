"""Claude provider implementation"""

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderError


class ClaudeProvider(BaseProvider):
    """Provider for Claude API"""

    def refine_prompt(
        self,
        prompt: str,
        focus_areas: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use Claude to refine the prompt"""
        try:
            import anthropic
        except ImportError:
            raise ProviderError("anthropic package not installed. Run: pip install anthropic")
        
        retry_attempts = self.config.advanced.retry_attempts
        
        for attempt in range(retry_attempts + 1):
            try:
                client = anthropic.Anthropic()
                
                # Build the refinement prompt
                system_prompt = "You are an expert at improving prompts for clarity and effectiveness."
                
                refinement_prompt = f"""Please refine the following prompt to make it clearer and more effective.

Original prompt: {prompt}

Focus areas: {', '.join(focus_areas) if focus_areas else 'clarity, specificity, actionability'}

Provide your response in JSON format with these fields:
- improved_prompt: The refined version of the prompt
- changes_made: Brief explanation of what was changed
- effectiveness_score: Rate the improvement (e.g., "8/10 - Much clearer")

Respond only with valid JSON."""
                
                response = client.messages.create(
                    model=f"claude-3-{self.config.provider.claude.get('model', 'opus')}-20240229",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": refinement_prompt}]
                )
                
                # Parse response
                result_text = response.model_dump_json()
                result_data = json.loads(result_text)
                
                # Extract the actual content
                if 'content' in result_data and len(result_data['content']) > 0:
                    content_text = result_data['content'][0].get('text', '{}')
                    return json.loads(content_text)
                
                return result_data
                
            except Exception as e:
                if attempt < retry_attempts:
                    continue
                raise ProviderError(f"Claude API error: {str(e)}")

    @staticmethod
    def is_available() -> bool:
        """Check if Claude API is available"""
        return os.getenv('ANTHROPIC_API_KEY') is not None