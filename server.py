"""
FastAPI бэкенд: AI-аналитика HSE
Роутер F-04: семантический поиск рекомендаций по мерам контроля
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Добавляем F-01 и F-04 в path чтобы импортировать engine
sys.path.insert(0, str(Path(__file__).parent / "F-01"))
import engine as f01_engine

sys.path.insert(0, str(Path(__file__).parent / "F-04"))
import engine as f04_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация всех AI-модулей при старте."""
    f01_engine.init()
    f04_engine.init()
    yield
    print("🛑 Сервер остановлен.")


app = FastAPI(
    title="AI HSE Analytics API",
    description="API для AI-модулей HSE-системы (Охрана Труда)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Схемы F-01 ───

class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Описание инцидента или опасной ситуации", min_length=5)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "Работник поскользнулся на разлитом масле возле насоса и ушиб колено"
            }]
        }
    }


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class ExtractedEntities(BaseModel):
    dates: list[str]
    locations: list[str]
    organizations: list[str]
    persons: list[str]


class ClassificationResponse(BaseModel):
    query: str
    predicted_class: str
    confidence: float
    probabilities: list[ClassProbability]
    entities: ExtractedEntities


# ─── Схемы F-04 ───

class RecommendationRequest(BaseModel):
    text: str = Field(..., description="Описание инцидента или опасной ситуации", min_length=3)
    top_k: int = Field(5, description="Количество рекомендаций (1-10)", ge=1, le=10)
    min_similarity: float = Field(
        40.0,
        description="Минимальный порог similarity (%). Результаты ниже — отбрасываются.",
        ge=0, le=100,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "Работник поскользнулся на разлитом масле возле насоса и ушиб колено",
                "top_k": 3,
                "min_similarity": 40,
            }]
        }
    }


class Recommendation(BaseModel):
    rank: int
    similarity: float
    recommendation: str
    corrective_measures: str | None = None
    preliminary_causes: str | None = None
    classification: str | None = None
    organization: str | None = None
    source_incident: str


class RecommendationResponse(BaseModel):
    query: str
    count: int
    warning: str | None = None
    recommendations: list[Recommendation]


# ─── Эндпоинты ───

@app.post("/f01/classify", response_model=ClassificationResponse, tags=["F-01"])
async def classify_incident(request: ClassificationRequest):
    """
    F-01: Автоматическая классификация происшествий по типу.
    Анализ текстового описания инцидента NLP-моделью.
    """
    result = f01_engine.classify(request.text)
    
    return ClassificationResponse(
        query=request.text,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        entities=result.get("entities", {"dates": [], "locations": [], "organizations": [], "persons": []}),
    )



@app.post("/f04/recommendations", response_model=RecommendationResponse, tags=["F-04"])
async def get_recommendations(request: RecommendationRequest):
    """
    F-04: Генерация рекомендаций по мерам контроля.
    Семантический поиск похожих инцидентов из истории (cosine kNN).
    """
    results = f04_engine.search(
        text=request.text,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )

    warning = None
    if not results:
        warning = (
            f"Не найдено похожих инцидентов с similarity ≥ {request.min_similarity}%. "
            f"Описанная ситуация не имеет аналогов в исторической базе."
        )

    return RecommendationResponse(
        query=request.text,
        count=len(results),
        warning=warning,
        recommendations=results,
    )


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "modules": {
            "f01": f01_engine.get_stats(),
            "f04": f04_engine.get_stats(),
        }
    }
