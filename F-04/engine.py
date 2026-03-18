"""
F-04 • NLP-движок: семантический поиск рекомендаций по мерам контроля.
Загружает модель, строит kNN-индекс, предоставляет функцию поиска.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent  # корень проекта (kmg/)

# ─── Глобальные объекты ───
_model = None
_nn_index = None
_df_incidents = None


def init():
    """Загрузка модели, данных и построение индекса. Вызывать один раз при старте."""
    global _model, _nn_index, _df_incidents

    print("⏳ [F-04] Загрузка данных...")
    df = pd.read_csv(DATA_DIR / 'Проишествия_clean.csv', sep=';')
    _df_incidents = df.dropna(subset=['Краткое_описание_происшествия', 'Рекомендации']).reset_index(drop=True)
    print(f"   ✅ {len(_df_incidents)} инцидентов с рекомендациями")

    print("⏳ [F-04] Загрузка NLP-модели (paraphrase-multilingual-MiniLM-L12-v2)...")
    _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✅ Модель загружена")

    print("⏳ [F-04] Векторизация инцидентов...")
    vectors = _model.encode(
        _df_incidents['Краткое_описание_происшествия'].tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    _nn_index = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    _nn_index.fit(vectors)
    print("   ✅ [F-04] Индекс готов!")


def search(text: str, top_k: int = 5, min_similarity: float = 40.0) -> list[dict]:
    """
    Семантический поиск похожих инцидентов.

    Args:
        text: описание инцидента / опасной ситуации
        top_k: сколько результатов вернуть
        min_similarity: минимальный порог similarity (%)

    Returns:
        list[dict] — список рекомендаций, отсортированных по similarity
    """
    if _model is None or _nn_index is None:
        raise RuntimeError("NLP-движок не инициализирован. Вызовите init() при старте.")

    fetch_k = min(top_k * 2, len(_df_incidents))
    query_vector = _model.encode([text])
    distances, indices = _nn_index.kneighbors(query_vector, n_neighbors=fetch_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = round((1 - dist) * 100, 1)

        if similarity < min_similarity:
            continue

        row = _df_incidents.iloc[idx]
        corrective = row.get('Корректирующие_меры', None)
        causes = row.get('Предварительные_причины', None)
        classification = row.get('Классификация_НС', None) or row.get('Классификация_ОМП', None)
        org = row.get('Наименование_организации_ДЗО', None)

        results.append({
            "rank": len(results) + 1,
            "similarity": similarity,
            "recommendation": str(row['Рекомендации']),
            "corrective_measures": str(corrective) if pd.notna(corrective) and str(corrective).strip() else None,
            "preliminary_causes": str(causes) if pd.notna(causes) and str(causes).strip() else None,
            "classification": str(classification) if pd.notna(classification) and str(classification).strip() else None,
            "organization": str(org) if pd.notna(org) and str(org).strip() else None,
            "source_incident": str(row['Краткое_описание_происшествия'])[:500],
        })

        if len(results) >= top_k:
            break

    return results


def get_stats() -> dict:
    """Статистика движка."""
    return {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "incidents_indexed": len(_df_incidents) if _df_incidents is not None else 0,
        "ready": _model is not None and _nn_index is not None,
    }
