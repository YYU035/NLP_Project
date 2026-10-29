"""Shared end-to-end entry point; wire modules here after implementation."""
from shared.contracts import Claim, PipelineResult


def run_pipeline(claim: Claim) -> PipelineResult:
    raise NotImplementedError("Integrate search, retrieval, ranking, verification and explanation modules.")
