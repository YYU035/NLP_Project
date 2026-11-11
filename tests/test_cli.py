"""Project entry points never silently provision weights or call live discovery."""
import pytest

from research_verifier import cli


def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(cli, "_run", lambda args: recorded.append(args) or 0)
    return recorded


def test_launch_only_starts_streamlit(monkeypatch):
    recorded = calls(monkeypatch)
    assert cli.main(["run", "--port", "8510"]) == 0
    assert len(recorded) == 1
    assert recorded[0][:3] == ["-m", "streamlit", "run"]
    assert recorded[0][-4:] == ["--server.address", "127.0.0.1", "--server.port", "8510"]


def test_recorded_smoke_does_not_implicitly_enable_live(monkeypatch):
    recorded = calls(monkeypatch)
    cli.main(["smoke", "--baseline", "--limit", "2"])
    assert recorded == [["-m", "research_verifier.runtime.smoke", "--limit", "2", "--baseline"]]


def test_child_process_keeps_resolved_paths_when_cwd_changes(monkeypatch, tmp_path):
    from types import SimpleNamespace

    root = tmp_path / "project"
    data = tmp_path / "model_storage"
    monkeypatch.setenv("RESEARCH_VERIFIER_ROOT", "relative/project")
    monkeypatch.setenv("RESEARCH_VERIFIER_DATA_DIR", "../model_storage")
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli, "DATA_DIR", data)
    captured = {}

    def run(arguments, *, cwd, env):
        captured.update(arguments=arguments, cwd=cwd, env=env)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._run(["-m", "research_verifier.runtime.setup_verifier"]) == 0
    assert captured["cwd"] == root
    assert captured["env"]["RESEARCH_VERIFIER_ROOT"] == str(root)
    assert captured["env"]["RESEARCH_VERIFIER_DATA_DIR"] == str(data)


def test_explicit_model_setup_is_ordered_and_stops_on_failure(monkeypatch):
    recorded = []
    def run(args):
        recorded.append(args)
        return 7 if len(recorded) == 2 else 0
    monkeypatch.setattr(cli, "_run", run)
    assert cli.main(["setup-models", "--language-model"]) == 7
    assert len(recorded) == 2
    assert recorded[0][1] == "paper_search.model_setup"
    assert recorded[1][1] == "research_verifier.runtime.setup_verifier"


@pytest.mark.parametrize("arguments", [["run", "--port", "0"], ["smoke", "--limit", "11"]])
def test_invalid_bounds_do_not_start_processes(monkeypatch, arguments):
    recorded = calls(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    assert recorded == []
