import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from catboost import CatBoostClassifier

print("1. Загружаем нашу умную базу данных...")
df = pd.read_csv("Dataset_for_Prediction_Semantic.csv", sep=";")

print("2. Подготавливаем Целевую переменную (Target)...")
# Сейчас в target_incidents лежит количество аварий. 
# Делаем так: 1 - риск есть (были аварии), 0 - риска нет (аварий не было)
df['risk_label'] = (df['target_incidents'] > 0).astype(int)

# 3. Разделяем данные на фичи (X) и таргет (y)
# Убираем колонки, которые не должны учить модель (имя организации и сами ответы)
X = df.drop(columns=['Организация', 'target_incidents', 'risk_label'])
y = df['risk_label']

print(f"Размер данных для обучения: {X.shape[1]} признаков!")

# 4. Разбиваем на обучающую и тестовую выборки (80% учим, 20% проверяем)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Начинаем обучение нейросети CatBoost...")
model = CatBoostClassifier(
    iterations=200,          # Количество попыток найти закономерности
    learning_rate=0.05,      # Скорость обучения
    depth=6,                 # Глубина анализа
    eval_metric='AUC',       # Метрика качества
    verbose=50               # Выводить лог каждые 50 шагов
)

# Обучаем!
model.fit(X_train, y_train, eval_set=(X_test, y_test))

print("\n4. Оцениваем качество предсказаний...")
preds_proba = model.predict_proba(X_test)[:, 1]
preds_class = model.predict(X_test)

# ROC-AUC > 0.7 - это уже отлично. Ближе к 1.0 - идеально.
print(f"ROC-AUC (Качество предсказания вероятности): {roc_auc_score(y_test, preds_proba):.4f}")

print("\n5. ДЕЛАЕМ ПРЕДИКТ ДЛЯ ДАШБОРДА (Для Жюри)...")
# Модель оценивает вероятность риска для ВСЕХ организаций на основе их поведения в "Коргау"
df['predicted_risk_probability'] = model.predict_proba(X)[:, 1]

# Создаем красивую итоговую таблицу
final_report = df[['Организация', 'predicted_risk_probability', 'target_incidents']].copy()
# Переводим в проценты для красоты
final_report['predicted_risk_probability'] = (final_report['predicted_risk_probability'] * 100).round(1)
# Сортируем от самых опасных к самым безопасным
final_report = final_report.sort_values(by='predicted_risk_probability', ascending=False)

final_report.to_csv("Final_Risk_Predictions.csv", sep=";", index=False)

print("\nУРА! ПРЕДИКТ ГОТОВ И СОХРАНЕН В 'Final_Risk_Predictions.csv'")
print("\n🔥 ТОП-3 ОРГАНИЗАЦИИ С МАКСИМАЛЬНЫМ РИСКОМ АВАРИИ В БУДУЩЕМ:")
print(final_report.head(3).to_string(index=False))
