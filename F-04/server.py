"""
F-04 • FastAPI бэкенд: Семантический поиск рекомендаций по мерам контроля
POST /recommendations — принимает описание инцидента, возвращает JSON с рекомендациями
"""

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from contextlib import asynccontextmanager

# ─── Глобальные объекты (инициализируются при старте) ───
model = None
nn_index = None
df_incidents = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели и данных при старте сервера."""
    global model, nn_index, df_incidents

    print("⏳ Загрузка данных...")
    df = pd.read_csv('../Проишествия_clean.csv', sep=';')
    df_incidents = df.dropna(subset=['Краткое_описание_происшествия', 'Рекомендации']).reset_index(drop=True)
    print(f"   ✅ {len(df_incidents)} инцидентов с рекомендациями")

    print("⏳ Загрузка NLP-модели (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✅ Модель загружена")

    print("⏳ Векторизация инцидентов...")
    vectors = model.encode(
        df_incidents['Краткое_описание_происшествия'].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    nn_index = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    nn_index.fit(vectors)
    print("   ✅ Индекс готов. Сервер запущен!")

    yield  # сервер работает

    print("🛑 Сервер остановлен.")


app = FastAPI(
    title="F-04 • AI Рекомендации HSE",
    description="Семантический поиск рекомендаций по мерам контроля на основе исторических инцидентов",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Схемы ───

MIN_SIMILARITY_DEFAULT = 40.0  # Порог отсечения по умолчанию (%)


class IncidentRequest(BaseModel):
    text: str = Field(..., description="Описание инцидента или опасной ситуации", min_length=3)
    top_k: int = Field(5, description="Количество рекомендаций (1-10)", ge=1, le=10)
    min_similarity: float = Field(
        MIN_SIMILARITY_DEFAULT,
        description="Минимальный порог similarity (%). Результаты ниже порога отбрасываются.",
        ge=0, le=100,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "text": "Работник поскользнулся на разлитом масле возле насоса и ушиб колено",
                "top_k": 3,
                "min_similarity": 60,
            }]
        }
    }


class Recommendation(BaseModel):
    rank: int
    similarity: float = Field(..., description="Cosine similarity (0-100%)")
    recommendation: str
    corrective_measures: str | None = None
    preliminary_causes: str | None = None
    classification: str | None = None
    organization: str | None = None
    source_incident: str = Field(..., description="Описание похожего исторического инцидента")


class RecommendationResponse(BaseModel):
    query: str
    count: int
    warning: str | None = Field(None, description="Предупреждение, если подходящих результатов не найдено")
    recommendations: list[Recommendation]


# ─── Эндпоинты ───

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: IncidentRequest):
    """
    Принимает описание инцидента и возвращает TOP-K рекомендаций
    из истории на основе семантического сходства (cosine similarity).
    Результаты с similarity ниже min_similarity отбрасываются.
    """
    # Ищем больше чем нужно, потом фильтруем по порогу
    fetch_k = min(request.top_k * 2, len(df_incidents))
    query_vector = model.encode([request.text])
    distances, indices = nn_index.kneighbors(query_vector, n_neighbors=fetch_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = round((1 - dist) * 100, 1)

        # Отсекаем по порогу
        if similarity < request.min_similarity:
            continue

        row = df_incidents.iloc[idx]
        corrective = row.get('Корректирующие_меры', None)
        causes = row.get('Предварительные_причины', None)
        classification = row.get('Классификация_НС', None) or row.get('Классификация_ОМП', None)
        org = row.get('Наименование_организации_ДЗО', None)

        results.append(Recommendation(
            rank=len(results) + 1,
            similarity=similarity,
            recommendation=str(row['Рекомендации']),
            corrective_measures=str(corrective) if pd.notna(corrective) and str(corrective).strip() else None,
            preliminary_causes=str(causes) if pd.notna(causes) and str(causes).strip() else None,
            classification=str(classification) if pd.notna(classification) and str(classification).strip() else None,
            organization=str(org) if pd.notna(org) and str(org).strip() else None,
            source_incident=str(row['Краткое_описание_происшествия'])[:500],
        ))

        if len(results) >= request.top_k:
            break

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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "incidents_indexed": len(df_incidents) if df_incidents is not None else 0,
    }
