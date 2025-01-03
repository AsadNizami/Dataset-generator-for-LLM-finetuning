from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, Any
from ..services.ollama_service import OllamaService
from fastapi.responses import StreamingResponse
import json
import asyncio

router = APIRouter()
ollama_service = OllamaService()

@router.get("/models")
async def get_models() -> Dict[str, Any]:
    """Get available models endpoint."""
    try:
        models = await ollama_service.get_models()
        return {"models": models}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/generate")
async def generate_response(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response from model endpoint."""
    required_fields = ['model', 'prompt']
    if not all(field in request for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        response = await ollama_service.generate_response(
            model=request['model'],
            prompt=request['prompt'],
            system_prompt=request.get('system_prompt'),
            temperature=request.get('temperature', 0.7)
        )
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate-dataset")
async def generate_dataset(
    file: UploadFile = File(...),
    num_pairs: int = Form(default=5),
    temperature: float = Form(default=0.7),
    model: str = Form(...),
    prompt: str = Form(...)
) -> StreamingResponse:
    """Generate Q&A dataset from uploaded file content."""
    try:
        content = await file.read()
        if len(content) > 1_000_000:  # 1MB limit
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 1MB"
            )

        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be a valid text file with UTF-8 encoding"
            )

        if not text_content.strip():
            raise HTTPException(
                status_code=400,
                detail="File content is empty"
            )

        if not 0 <= temperature <= 1:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be between 0 and 1"
            )

        async def generate():
            try:
                async for pair in ollama_service.generate_dataset(
                    content=text_content,
                    num_pairs=num_pairs,
                    temperature=temperature,
                    model=model,
                    prompt=prompt
                ):
                    data = {
                        "conversations": [
                            {"from": "human", "value": pair["question"]},
                            {"from": "assistant", "value": pair["answer"]}
                        ],
                        "source": file.filename
                    }
                    yield json.dumps(data) + "\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset generation failed: {str(e)}"
        ) 