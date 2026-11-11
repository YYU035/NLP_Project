import hashlib
import io

import pytest

import research_verifier.runtime.setup_language_model as local_model_setup


def test_public_weight_download_is_activated_only_after_hash_and_size_match(tmp_path, monkeypatch):
    content = b"verified weights"
    monkeypatch.setattr(local_model_setup, "MODEL_WEIGHTS_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(local_model_setup, "MODEL_WEIGHTS_SIZE", len(content))
    calls = []
    def opener(request, **kwargs):
        calls.append(request)
        return io.BytesIO(content)
    monkeypatch.setattr(local_model_setup, "urlopen", opener)
    destination = tmp_path / "model.safetensors"
    local_model_setup.download_weights(destination)
    assert destination.read_bytes() == content
    assert len(calls) == 1
    local_model_setup.download_weights(destination)
    assert len(calls) == 1  # existing verified weights require no request
    destination.unlink()
    monkeypatch.setattr(local_model_setup, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum"):
        local_model_setup.download_weights(destination)
    assert not destination.exists()
