"""
F-01 • NLP-движок: Автоматическая классификация происшествий по типу на основе текстового описания.
Использует kNN по эмбеддингам sentence-transformers для взвешенного голосования классов.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent  # корень проекта (kmg/)

# ─── Глобальные объекты ───
_model = None
_nn_index = None
_df_incidents = None


def init():
    """Загрузка модели, данных и построение индекса классификации."""
    global _model, _nn_index, _df_incidents

    print("⏳ [F-01] Загрузка данных для классификации...")
    df = pd.read_csv(DATA_DIR / 'Проишествия_clean.csv', sep=';')
    
    # Объединяем колонки классификации в один целевой класс
    df['target_class'] = df['Классификация_НС'].fillna(df['Классификация_ОМП'])
    _df_incidents = df.dropna(subset=['Краткое_описание_происшествия', 'target_class']).reset_index(drop=True)
    print(f"   ✅ [F-01] {len(_df_incidents)} инцидентов с известным классом")

    print("⏳ [F-01] Загрузка NLP-модели (paraphrase-multilingual-MiniLM-L12-v2)...")
    _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("⏳ [F-01] Векторизация обучающей выборки...")
    vectors = _model.encode(
        _df_incidents['Краткое_описание_происшествия'].tolist(),
        show_progress_bar=False,
        batch_size=64,
    )
    # n_neighbors=15 для мягкого голосования вероятностей
    _nn_index = NearestNeighbors(n_neighbors=15, metric='cosine', algorithm='brute')
    _nn_index.fit(vectors)
    print("   ✅ [F-01] Индекс классификатора готов!")


def classify(text: str) -> dict:
    """
    Классифицирует текстовое описание инцидента.
    Возвращает словарь с предсказанным классом и вероятностями топ-классов.
    """
    if _model is None or _nn_index is None:
        raise RuntimeError("NLP-движок F-01 не инициализирован.")

    query_vector = _model.encode([text])
    distances, indices = _nn_index.kneighbors(query_vector)

    class_scores = defaultdict(float)
    total_score = 0.0

    # Взвешенное голосование: чем меньше дистанция, тем больше вес
    # Вес = 1 / (дистанция + 0.01) чтобы избежать деления на ноль
    for idx, dist in zip(indices[0], distances[0]):
        row = _df_incidents.iloc[idx]
        incident_class = str(row['target_class']).strip()
        
        weight = 1.0 / (dist + 0.01)
        class_scores[incident_class] += weight
        total_score += weight

    # Решаем вероятности
    probabilities = []
    for cls, score in class_scores.items():
        prob = (score / total_score) * 100
        probabilities.append({"class_name": cls, "probability": round(prob, 1)})

    # Сортируем по убыванию вероятности
    probabilities.sort(key=lambda x: x["probability"], reverse=True)
    
    predicted_class = probabilities[0]["class_name"] if probabilities else "Неизвестно"
    confidence = probabilities[0]["probability"] if probabilities else 0.0

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities[:5]  # Отдаём только топ-5 вероятностей
    }


def get_stats() -> dict:
    return {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "incidents_indexed": len(_df_incidents) if _df_incidents is not None else 0,
        "ready": _model is not None and _nn_index is not None,
    }
