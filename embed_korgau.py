import os
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_CSV = "korgau_prepared.csv"
OUTPUT_CSV = "korgau_with_embeddings.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Файл не найден: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "signal_text" not in df.columns:
        raise KeyError("В файле нет колонки 'signal_text'")

    texts = df["signal_text"].fillna("").astype(str).tolist()

    print("Загружаем модель...")
    model = SentenceTransformer(MODEL_NAME)

    print("Строим embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb_df = pd.DataFrame(embeddings)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

    result = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nГотово. Сохранен файл: {OUTPUT_CSV}")
    print("Размер:", result.shape)


if __name__ == "__main__":
    main()