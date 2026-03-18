import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import time

print("1. Загрузка данных...")
df = pd.read_csv('Проишествия_merged_dates.csv', sep=';')

# Оставляем только те строки, где есть и описание, и рекомендации
df = df.dropna(subset=['Краткое_описание_происшествия', 'Рекомендации']).reset_index(drop=True)
print(f"Готово! Найдено {len(df)} записей с рекомендациями.")

print("2. Загрузка NLP-модели...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("3. Векторизация исторических инцидентов и сборка индекса...")
start_time = time.time()

# Превращаем все тексты инцидентов в векторы (матрица чисел)
# С sentence-transformers это можно сделать одной строкой (он сам оптимизирует батчи)
vectors = model.encode(df['Краткое_описание_происшествия'].tolist(), show_progress_bar=True)

# Используем стабильный NearestNeighbors из scikit-learn (метрика косинусного расстояния)
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
nn_model.fit(vectors)

print(f"Векторная база собрана за {time.time() - start_time:.2f} секунд!")

# =====================================================================
# ИМИТАЦИЯ РАБОТЫ API В РЕАЛЬНОМ ВРЕМЕНИ
# =====================================================================

print("\n--- ТЕСТ СИСТЕМЫ ---")
new_alert_text = "Работник поскользнулся на разлитом масле возле насоса и ушиб колено."
print(f"Новый алерт: {new_alert_text}\n")

# 1. Векторизуем новый текст (передаем в виде списка)
new_vector = model.encode([new_alert_text])

# 2. Ищем ТОП-3 самых похожих старых инцидента
distances, indices = nn_model.kneighbors(new_vector)

print("✅ ИИ сгенерировал 3 рекомендации на основе исторического опыта:\n")
# indices[0] содержит индексы трех самых похожих строк
for rank, idx in enumerate(indices[0]):
    past_recommendation = df.iloc[idx]['Рекомендации']
    past_incident = df.iloc[idx]['Краткое_описание_происшествия']

    print(f"💡 Рекомендация #{rank + 1}:")
    print(f"{past_recommendation}")
    print(f"(Основано на прошлом инциденте: '{past_incident[:100]}...')\n")