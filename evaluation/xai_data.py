"""Explicitly stage checksum-pinned public SciFact train/dev/corpus data."""
from evaluation.datasets import SCIFACT_DIRECTORY, SCIFACT_SHA256, stage_scifact

EXPECTED = SCIFACT_SHA256


def main():
    stage_scifact()
    print(f"Staged verified public SciFact data: {SCIFACT_DIRECTORY}")


if __name__ == "__main__":
    main()
