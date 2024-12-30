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
        model: str = "llama3.2",
        prompt: str = None
    ) -> List[Dict[str, str]]:
        """Generate Q&A pairs dataset from content."""
        if not await self.check_model_availability(model):
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' is not available. Please run 'ollama pull {model}' first."
            )

        invalid_responses = 0
        total_attempts = 0

        # Update default prompt
        default_prompt = """Analyze the given text and create exactly one question-answer pair.

You must:
1. Return only a JSON array containing one object
2. Use exactly this format, no extra text:
[
    {
        "question": "Clear, specific question from the text?",
        "answer": "Direct, factual answer from the text."
    }
]

Important:
- Keep answers concise and factual
- Questions should be specific and answerable from the text
- Do not add any explanations or additional text
- Do not create multiple pairs
- Ensure valid JSON syntax with double quotes"""
        
        instruction = prompt if prompt else default_prompt

        # Generate one pair at a time
        for i in range(num_pairs):
            max_attempts = 3  # Maximum attempts per pair
            attempts = 0
            
            while attempts < max_attempts:
                total_attempts += 1
                attempts += 1
                
                full_prompt = f"""{instruction}

                Text to process:
                {content}"""

                try:
                    response = await self.generate_response(
                        model=model,
                        prompt=full_prompt,
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
                                print(f"✓ Generated valid pair {i+1}/{num_pairs}")
                                yield {
                                    "question": str(pair['question']).strip(),
                                    "answer": str(pair['answer']).strip()
                                }
                                break  # Success, move to next pair
                        else:
                            print(f"✗ Invalid response format: {cleaned_text}...")
                            invalid_responses += 1
                    except json.JSONDecodeError:
                        print(f"✗ JSON parse error: {cleaned_text[:100]}...")
                        invalid_responses += 1
                        
                except Exception as e:
                    print(f"✗ Error generating pair: {str(e)}")
                    invalid_responses += 1

                if attempts == max_attempts:
                    print(f"! Max attempts reached for pair {i+1}")
                    invalid_responses += 1

        print("\n=== Generation Statistics ===")
        print(f"Total attempts: {total_attempts}")
        print(f"Invalid responses: {invalid_responses}")
        print(f"Success rate: {((total_attempts - invalid_responses) / total_attempts * 100):.1f}%")
        print("=========================\n")