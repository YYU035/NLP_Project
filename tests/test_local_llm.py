import json
from pathlib import Path

import pytest

from research_verifier.explanations.language_model import LocalTransformersClient, LOCAL_MODEL_ID, LOCAL_MODEL_REVISION


def model_dir(tmp_path):
    path = tmp_path / "weights"
    path.mkdir()
    (path / "pinned_revision.txt").write_text(LOCAL_MODEL_REVISION)
    (path / "config.json").write_text("{}")
    (path / "LICENSE").write_text("Apache License\nVersion 2.0")
    return path


def test_local_client_requires_explicitly_downloaded_pinned_weights(tmp_path):
    with pytest.raises(ValueError, match="setup-local-llm"):
        LocalTransformersClient(tmp_path / "missing")
    directory = model_dir(tmp_path)
    (directory / "pinned_revision.txt").write_text("wrong")
    with pytest.raises(ValueError): LocalTransformersClient(directory)


def test_local_configuration_needs_no_account_or_key(tmp_path, monkeypatch):
    directory = model_dir(tmp_path)
    monkeypatch.setenv("EXPLANATION_LOCAL_MODEL_PATH", str(directory))
    client = LocalTransformersClient.from_env()
    assert client.model.startswith(LOCAL_MODEL_ID)
    assert client.settings["backend"] == "local_transformers"
    assert client.settings["max_attempts"] == 1


def test_local_generation_uses_only_local_files_and_does_not_silently_truncate(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    loaded = []
    class Tokenizer:
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        def decode(self, tokens, **kwargs): return '{"verdict":"INSUFFICIENT","statements":[],"correction":null}'
    class Model:
        def eval(self): return self
        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 4, 2]])
    def load_tokenizer(path, **kwargs):
        loaded.append(kwargs)
        return Tokenizer()
    def load_model(path, **kwargs):
        loaded.append(kwargs)
        return Model()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    client = LocalTransformersClient(model_dir(tmp_path), max_context_tokens=4)
    assert json.loads(client.complete([{"role": "user", "content": "exact claim"}]))["verdict"] == "INSUFFICIENT"
    assert all(x["local_files_only"] and x["trust_remote_code"] is False for x in loaded)
    small = LocalTransformersClient(tmp_path / "weights", max_context_tokens=2)
    with pytest.raises(ValueError, match="context"):
        small.complete([{"role": "user", "content": "oversized"}])


def test_live_catalog_configuration_cannot_use_paid_account_credentials(monkeypatch, tmp_path):
    from research_verifier.discovery.service import _service_factory
    pytest.importorskip("paper_search.retrieval")
    import paper_search.retrieval
    seen = {}
    monkeypatch.setenv("EXPLANATION_DISCOVERY_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("OPENALEX_API_KEY", "must-not-be-used")
    monkeypatch.setenv("OPENAIRE_TOKEN", "must-not-be-used")
    def factory(**kwargs): seen.update(kwargs); return object()
    monkeypatch.setattr(paper_search.retrieval, "RetrievalService", factory)
    _service_factory()
    assert seen["settings"].openalex_api_key is None
    assert seen["settings"].openaire_token is None
    assert seen["settings"].semantic_scholar_enabled is False
