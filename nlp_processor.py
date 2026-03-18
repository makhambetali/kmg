from sentence_transformers import SentenceTransformer


class SemanticNLPProcessor:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]):
        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )