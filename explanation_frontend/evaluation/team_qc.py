"""Run pinned teammate tests in isolated module-local workspaces, without network."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCES = {
    "raghavi": ("259d2a9e90fb8e9d5988f030a156733f4f3576d1", "live_search"),
    "naila": ("2bc7d81292313498fd70dd13656024d00cf3dda4", "retrieval"),
    "yuyue": ("a91e257ab260ded8df8c2cc2f214c4c7fd16af6c", "ranking_evaluation"),
    "yuyue_current": ("90aaca109585c12c877367b1a08fe4dfee4a4de3", "ranking_evaluation"),
    "chong_yao": ("1ac06b675e3d0719a63cb8efe98eda4f2c5fd576", "claim_verifier"),
}


def run(owner, portability_fix=False, stable_cutoff_fix=False):
    revision, folder = SOURCES[owner]
    workspace = ROOT / ".tmp" / "team_qc" / owner
    workspace.mkdir(parents=True, exist_ok=True)
    prior_probe = workspace / folder / 'tests' / 'test_qc_cutoff.py'
    if prior_probe.is_file():
        prior_probe.unlink()  # Only our known harness file, never an owner source.
    folders = [folder]
    if owner == 'yuyue_current':
        folders += ['shared','retrieval','live_search/src','claim_verifier','README.md']
    archive = subprocess.check_output(["git", "archive", revision, *folders], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            target = (workspace / member.name).resolve()
            if not target.is_relative_to(workspace.resolve()):
                raise ValueError("Archive path escapes workspace")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as handle:
                    target.write_bytes(handle.read())
    if portability_fix:
        if owner != "yuyue":
            raise ValueError("Portability patch is only defined for Yu Yue")
        file = workspace / folder / "evaluate.py"
        old = file.read_text(encoding="utf-8")
        new = old.replace("str(path.relative_to(root))", "path.relative_to(root).as_posix()")
        new = new.replace("str(path.relative_to(args.model_path))", "path.relative_to(args.model_path).as_posix()")
        file.write_text(new, encoding="utf-8")
        import difflib
        patches = ROOT / "qc_suggestions"
        patches.mkdir(exist_ok=True)
        (patches / "yuyue-windows-provenance.patch").write_text("".join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), fromfile="a/ranking_evaluation/evaluate.py",
            tofile="b/ranking_evaluation/evaluate.py")), encoding="utf-8")
    if stable_cutoff_fix:
        if owner != "naila":
            raise ValueError("Stable cutoff patch is only defined for Naila")
        file = workspace / folder / "retriever.py"
        text = file.read_text(encoding="utf-8")
        text = text.replace("idx = np.argpartition(-scores, k - 1)[:k]\n    return idx[np.lexsort((idx, -scores[idx]))]",
                            "return np.lexsort((np.arange(len(scores)), -scores))[:k]")
        file.write_text(text, encoding="utf-8")
        (workspace / folder / 'tests' / 'test_qc_cutoff.py').write_text(
            'import numpy as np\nfrom retrieval.retriever import _top_k\n'
            'def test_equal_scores_use_lowest_indices_at_cutoff():\n'
            '    assert _top_k(np.array([0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]),3).tolist() == [4,0,1]\n',
            encoding='utf-8')
    # Test copies remain unchanged. This harness adds only runtime network guards
    # and an explicit offline replacement for Person 4's download-capable init.
    guard = '''import socket
_connect = socket.socket.connect
_create_connection = socket.create_connection
def guarded_connect(self, address):
    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
        return _connect(self, address)
    raise RuntimeError("Team QC forbids external network requests")
def guarded_create(address, *args, **kwargs):
    if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
        return _create_connection(address, *args, **kwargs)
    raise RuntimeError("Team QC forbids external network requests")
socket.socket.connect = guarded_connect
socket.create_connection = guarded_create
'''
    if owner == "chong_yao":
        guard += '''
import verifier as module
def offline_init(self, checkpoint=None, revision=None, tokenizer_max_length=512, threshold=.50, seed=42, device=None):
    from explanation_frontend.pipeline import local_verifier
    self.__dict__.update(local_verifier().__dict__)
    self.threshold, self.max_length, self.seed = threshold, tokenizer_max_length, seed
    self.label_mapping = self._build_label_mapping()
module.ClaimVerifier.__init__ = offline_init
'''
    (workspace / "conftest.py").write_text(guard, encoding="utf-8")
    (workspace / folder / "conftest.py").write_text(guard, encoding="utf-8")
    env = os.environ.copy()
    for key in list(env):
        if key.endswith(("API_KEY", "ACCESS_TOKEN")) or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT / ".model-cache" / "huggingface"),
               TORCH_HOME=str(ROOT / ".model-cache" / "torch"),
               PYTHONPATH=os.pathsep.join([str(workspace), str(workspace / folder),
                    str(workspace / folder / "src"), str(REPO)]))
    reports = ROOT / "outputs" / "team_qc"
    reports.mkdir(parents=True, exist_ok=True)
    name = owner + ("-portability-patched" if portability_fix else "") + ("-stable-cutoff-patched" if stable_cutoff_fix else "")
    xml_path = reports / f"{name}.xml"
    tests = workspace / folder / ("test_verifier.py" if owner == "chong_yao" else "tests")
    result = subprocess.run([sys.executable, "-m", "pytest", str(tests), "-q",
        "-o", f"cache_dir={workspace / '.pytest_cache'}", "--basetemp", str(workspace / ".pytest_tmp"),
        "--junitxml", str(xml_path)], cwd=workspace / folder if owner == "raghavi" else workspace,
        env=env, capture_output=True, text=True, encoding="utf-8")
    (reports / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    summary = {"owner": owner, "revision": revision, "exit_code": result.returncode,
               "network": "external blocked; loopback allowed for asyncio", "initialization_adapter": owner == "chong_yao",
               "scratch_portability_patch": portability_fix, "scratch_stable_cutoff_patch": stable_cutoff_fix}
    if xml_path.is_file():
        cases = ET.parse(xml_path).findall(".//testcase")
        summary.update(total=len(cases), failures=sum(x.find("failure") is not None for x in cases),
            errors=sum(x.find("error") is not None for x in cases),
            skipped=sum(x.find("skipped") is not None for x in cases))
        summary["passed"] = summary["total"] - sum(summary[k] for k in ["failures", "errors", "skipped"])
    (reports / f"{name}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("owner", choices=SOURCES)
    parser.add_argument("--portability-fix", action="store_true")
    parser.add_argument("--stable-cutoff-fix", action="store_true")
    args = parser.parse_args()
    sys.exit(run(args.owner, args.portability_fix, args.stable_cutoff_fix))
