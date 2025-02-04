from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Dict, Any
from ..services.ollama_service import OllamaService
from ..services.rag import query_rag, load_documents, split_documents, add_to_chroma
from fastapi.responses import StreamingResponse
import json
import os

router = APIRouter()
ollama_service = OllamaService()

@router.get("/models")
async def get_models() -> Dict[str, Any]:
    """Get available models endpoint."""
    models = await ollama_service.get_models()
    return {"models": models}

@router.post("/generate")
async def generate_response(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response from model endpoint."""
    required_fields = ['model', 'prompt']
    if not all(field in request for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    response = await ollama_service.generate_response(
        model=request['model'],
        prompt=request['prompt'],
        system_prompt=request.get('system_prompt'),
        temperature=request.get('temperature', 0.7)
    )

    return response

@router.post("/generate-dataset")
async def generate_dataset(
    file: UploadFile = File(...),
    num_pairs: int = Form(default=0),
    temperature: float = Form(default=0.7),
    model: str = Form(...),
    prompt: str = Form(...),
    keywordprompt: str = Form(...)
) -> StreamingResponse:
    """Generate Q&A dataset from uploaded file content."""
    print(f'{num_pairs=}, {keywordprompt=}')
    
    try:
        content = await file.read()
        if len(content) > 5_000_000: 
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 5MB"
            )
        os.makedirs("temp", exist_ok=True) 

        file_location = f"./temp/{file.filename}"  # Specify your desired path
        with open(file_location, "wb") as f:
            f.write(content)

        print(f"File saved to: {file_location}")

        loader = load_documents('./temp')
        chunks = split_documents(loader)
        add_to_chroma(chunks, model)
        os.remove(file_location)


        if not 0 <= temperature <= 1:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be between 0 and 1"
            )

        async def generate():
            # If num_pairs is zero, yield an empty response
            if num_pairs == 0:
                yield json.dumps({"conversations": [], "source": file.filename}) + "\n"
                return

            try:
                async for pair in ollama_service.generate_dataset(
                    num_pairs=num_pairs,
                    temperature=temperature,
                    model=model,
                    prompt=prompt,
                    keywordprompt=keywordprompt
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