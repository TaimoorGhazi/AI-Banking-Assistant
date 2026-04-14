from src.AI_Banking_Assistant.retrieval import embeddings


def test_generate_embedding_uses_normalized_vectors(monkeypatch):
	class FakeModel:
		def __init__(self):
			self.calls = []

		def encode(self, *args, **kwargs):
			self.calls.append(kwargs)
			return [0.1, 0.2, 0.3]

	fake_model = FakeModel()
	monkeypatch.setattr(embeddings, "load_embedding_model", lambda model_name=None: fake_model)

	vector = embeddings.generate_embedding("sample query")

	assert vector.shape == (3,)
	assert fake_model.calls, "Expected encode() to be called"
	assert fake_model.calls[0].get("normalize_embeddings") is True
