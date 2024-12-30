from typing import Dict, Optional, Any, List
import httpx
from fastapi import HTTPException
import json

class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.MODEL_NAME = "llama3.2:latest"
        print(f"OllamaService initialized with base_url: {base_url}")  # Debug log

    async def check_model_availability(self, model_name: str) -> bool:
        """Check if specified model is available."""
        try:
            models = await self.get_models()
            print(f"Available models: {models}")  # Debug log
            is_available = any(model['name'] == model_name for model in models)
            print(f"Is {model_name} available? {is_available}")  # Debug log
            return is_available
        except Exception as e:
            print(f"Error checking model availability: {str(e)}")  # Debug log
            return False

    async def get_models(self) -> list:
        """Get available models from Ollama."""
        try:
            print("Attempting to get models from Ollama...")  # Debug log
            response = await self.client.get(f"{self.base_url}/api/tags")
            print(f"Response status: {response.status_code}")  # Debug log
            print(f"Response content: {response.text}")  # Debug log
            
            models = response.json().get('models', [])
            processed_models = [
                {'name': model['name'], 'modified_at': model.get('modified_at')}
                for model in models
                if 'name' in model
            ]
            print(f"Processed models: {processed_models}")  # Debug log
            return processed_models
        except Exception as e:
            print(f"Error getting models: {str(e)}")  # Debug log
            raise HTTPException(
                status_code=503,
                detail=f"Ollama service unavailable: {str(e)}"
            )

    async def generate_response(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate a response from the specified model."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Generation failed: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def generate_dataset(
        self,
        content: str,
        num_pairs: int = 5,
        temperature: float = 0.7,
        model: str = "llama3.2"
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs dataset from content."""
        if not await self.check_model_availability(model):
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' is not available. Please run 'ollama pull {model}' first."
            )

        # Generate one pair at a time
        for i in range(num_pairs):
            prompt = f"""Generate a single question-answer pair from the following text.
            Return your response in this exact format, with no additional text:
            [
                {{"question":"Your question here?","answer":"Your answer here."}}
            ]

            Text to process:
            {content}"""

            try:
                response = await self.generate_response(
                    model=model,
                    prompt=prompt,
                    temperature=temperature
                )
                
                response_text = response.get('response', '')
                if not response_text:
                    continue

                # Clean and parse the response
                cleaned_text = response_text.strip().replace('\n', '').replace('    ', '')
                try:
                    pairs = json.loads(cleaned_text)
                    if isinstance(pairs, list) and len(pairs) > 0:
                        pair = pairs[0]
                        if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                            yield {
                                "question": str(pair['question']).strip(),
                                "answer": str(pair['answer']).strip()
                            }
                except json.JSONDecodeError:
                    continue
                
            except Exception as e:
                print(f"Error generating pair {i+1}: {str(e)}")
                continue