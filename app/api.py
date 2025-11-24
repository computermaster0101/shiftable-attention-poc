from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .model_manager import model_manager

app = FastAPI(title="Shiftable Multi-Agent PoC API", version="1.1.0")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt.")
    max_new_tokens: int = Field(
        64,
        description="Maximum number of new tokens to generate.",
        ge=1,
        le=512,
    )
    temperature: float = Field(
        1.0,
        description="Softmax temperature. 0 means greedy decoding.",
        ge=0.0,
    )
    top_k: int = Field(
        50,
        description="Top-k sampling. 0 means no top-k filter.",
        ge=0,
    )


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    specialists: List[str]


class AddSpecialistRequest(BaseModel):
    name: str = Field(..., description="Name of the new specialist. Must match data/<name> directory.")


class AddSpecialistResponse(BaseModel):
    message: str
    specialists: List[str]


class HealthResponse(BaseModel):
    initialized: bool
    device: str
    tokenizer_path: str
    generalist_ckpt_path: str
    shiftable_ckpt_path: str
    specialists: List[str]


class DeleteSpecialistResponse(BaseModel):
    message: str
    specialists: List[str]


@app.on_event("startup")
def on_startup() -> None:
    """
    Application startup hook.

    Performs initialization:
    - Train generalist and shiftable models if needed.
    - Load tokenizer and shiftable model into memory.
    """
    try:
        model_manager.ensure_initialized()
    except Exception as e:
        # We don't crash the app on startup, but health endpoint will show the failure
        import logging

        logging.getLogger(__name__).exception("Error during startup initialization: %s", e)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """
    Serve the simple UI to chat with the model and manage specialists.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend index.html not found.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Check initialization status and list available specialists.
    """
    try:
        status = model_manager.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return HealthResponse(
        initialized=status.get("initialized", False),
        device=status.get("device", "unknown"),
        tokenizer_path=status.get("tokenizer_path", ""),
        generalist_ckpt_path=status.get("generalist_ckpt_path", ""),
        shiftable_ckpt_path=status.get("shiftable_ckpt_path", ""),
        specialists=status.get("specialists", []),
    )


@app.get("/specialists", response_model=List[str])
def list_specialists() -> List[str]:
    """
    List the currently configured specialists.
    """
    try:
        return model_manager.list_specialists()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Generate text from the current shiftable model (generalist + specialists).
    """
    try:
        result = model_manager.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        specialists = model_manager.list_specialists()
        return GenerateResponse(
            prompt=result["prompt"],
            completion=result["completion"],
            full_text=result["full_text"],
            specialists=specialists,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/specialists", response_model=AddSpecialistResponse)
def add_specialist(req: AddSpecialistRequest) -> AddSpecialistResponse:
    """
    Add a new specialist domain.

    Process:
    1. You create a corpus directory:
         shiftable_project/data/<name>/*.txt
    2. Call this endpoint with JSON body: { "name": "<name>" }.
    3. The API retrains the shiftable model over all specialists (including the new one).
    """
    try:
        specialists = model_manager.add_specialist(req.name)
        return AddSpecialistResponse(
            message=f"Specialist '{req.name}' is now configured.",
            specialists=specialists,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/specialists/{name}", response_model=DeleteSpecialistResponse)
def delete_specialist(name: str) -> DeleteSpecialistResponse:
    """
    Delete an existing specialist.

    Process:
    - Removes its corpus directory at shiftable_project/data/<name>
    - Retrains the shiftable model over remaining specialists

    At least one specialist must remain.
    """
    try:
        specialists = model_manager.delete_specialist(name)
        return DeleteSpecialistResponse(
            message=f"Specialist '{name}' has been deleted.",
            specialists=specialists,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
