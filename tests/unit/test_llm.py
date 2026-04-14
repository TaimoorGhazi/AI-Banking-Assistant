from src.AI_Banking_Assistant.core.config import Config
from src.AI_Banking_Assistant.llm import inference
from src.AI_Banking_Assistant.llm.prompt_templates import format_rag_prompt


def test_generate_response_falls_back_when_model_unavailable(monkeypatch):
	monkeypatch.setenv("USE_GROQ", "false")
	Config._instance = None

	def raise_on_get_model():
		raise RuntimeError("model not available")

	monkeypatch.setattr(inference, "get_model", raise_on_get_model)

	response = inference.generate_response("Customer Question: What is a savings account?")

	assert "local fallback mode" in response
	assert "What is a savings account?" in response
	Config._instance = None


def test_generate_chat_response_falls_back_when_chat_template_fails(monkeypatch):
	monkeypatch.setenv("USE_GROQ", "false")
	Config._instance = None

	class BrokenTokenizer:
		def apply_chat_template(self, *args, **kwargs):
			raise RuntimeError("template unavailable")

	def fake_get_model():
		return object(), BrokenTokenizer()

	monkeypatch.setattr(inference, "get_model", fake_get_model)

	response = inference.generate_chat_response([
		{"role": "user", "content": "How do transfers work?"},
	])

	assert "local fallback mode" in response
	assert "How do transfers work?" in response
	Config._instance = None


def test_runtime_profile_defaults_to_cpu_local_when_invalid_mode(monkeypatch):
	monkeypatch.setenv("USE_GROQ", "false")
	Config._instance = None
	monkeypatch.setenv("RUNTIME_MODE", "invalid_mode")

	config = Config()

	assert config.runtime_mode == "cpu_local"
	assert isinstance(config.runtime_profile, dict)

	Config._instance = None


def test_generate_response_passes_max_time_to_model(monkeypatch):
	monkeypatch.setenv("USE_GROQ", "false")
	Config._instance = None

	class FakeTensor:
		def __init__(self, values):
			self.values = values
			self.shape = (1, len(values))

		def to(self, _device):
			return self

		def __getitem__(self, key):
			if isinstance(key, slice):
				return self.values[key]
			return self.values[key]

	class FakeTokenizer:
		pad_token_id = 0
		eos_token_id = 2

		def __call__(self, *_args, **_kwargs):
			return {
				"input_ids": FakeTensor([10, 11]),
				"attention_mask": FakeTensor([1, 1]),
			}

		def decode(self, _tokens, **_kwargs):
			return "ok"

	class FakeModel:
		device = "cpu"

		def __init__(self):
			self.kwargs = None

		def generate(self, **kwargs):
			self.kwargs = kwargs
			return [FakeTensor([10, 11, 12])]

	class FakeNoGrad:
		def __enter__(self):
			return None

		def __exit__(self, exc_type, exc, tb):
			return False

	class FakeTorch:
		@staticmethod
		def no_grad():
			return FakeNoGrad()

	fake_model = FakeModel()
	fake_tokenizer = FakeTokenizer()
	monkeypatch.setattr(inference, "get_model", lambda: (fake_model, fake_tokenizer))

	import sys
	monkeypatch.setitem(sys.modules, "torch", FakeTorch)

	Config._instance = None
	response = inference.generate_response("Question: test")

	assert response == "ok"
	assert "max_time" in fake_model.kwargs
	assert fake_model.kwargs["max_time"] == Config().generation_max_time

	Config._instance = None


def test_rag_prompt_prefers_answering_when_context_has_relevant_info():
	prompt = format_rag_prompt(
		query="How can I block my card if it is lost?",
		context="If your debit or credit card is lost or stolen: immediately call 1-800-CARD-STOP.",
	)

	assert "If there is any relevant information in the context" in prompt
	assert "Do not use the fallback when the context contains partial but useful details" in prompt
