"""Audit installed research datasets without model inference or network calls."""

import argparse
from collections import Counter
import json
from pathlib import Path

from shared.datasets import csfcube, matter_of_fact


def audit(data_root: Path) -> dict:
    cs = csfcube.load(data_root / "csfcube", split="all")
    cs_counts = {facet: sum(q["facet"] == facet for q in cs["queries"].values())
                 for facet in csfcube.FACETS}
    self_matches = {}
    for facet in csfcube.FACETS:
        original = json.loads((data_root / "csfcube" / f"test-pid2anns-csfcube-{facet}.json").read_text())
        self_matches[facet] = sum(pid in row["cands"] for pid, row in original.items())
    report = {"type": "dataset loading/protocol audit, not a model quality evaluation",
        "active_selection": ["SciFact", "CSFCube", "RECIPER"],
        "csfcube": {"documents": len(cs["documents"]),
            "query_facet_tasks": len(cs["queries"]), "facets": cs_counts,
            "fold1_dev_tasks": len(csfcube.load(data_root / "csfcube")["queries"]),
            "raw_self_matches_excluded": self_matches,
            "grades": dict(Counter(str(g) for row in cs["grades"].values() for g in row.values())),
            "manifest": cs["manifest"]}}
    if (data_root / 'reciper').exists():
        from shared.datasets.reciper import load as load_reciper
        rec = load_reciper(data_root / 'reciper')
        report['reciper'] = {'papers': len(rec['documents']), 'questions': len(rec['queries']),
            'source_paper_targets': len(rec['qrels']), 'version': rec['version'],
            'protocol': rec['protocol'], 'manifest': rec['manifest']}
    else:
        report['reciper'] = {'status': 'not_installed'}
    if (data_root / 'climatecheck').exists():
        from shared.datasets.climatecheck import load
        splits = {}
        for split in ('train', 'test'):
            dataset = load(data_root / 'climatecheck', split=split)
            splits[split] = {'exact_claims': len(dataset['queries']),
                'abstracts': len(dataset['documents']),
                'annotated_pairs': sum(len(row) for row in dataset['grades'].values()),
                'labels': dict(Counter(label for row in dataset['pair_labels'].values() for label in row.values())),
                'no_positive_claims': sum(not any(g > 0 for g in row.values()) for row in dataset['grades'].values()),
                'protocol': dataset['protocol'], 'version': dataset['version']}
        report['climatecheck'] = {'role': 'archived experiment; subject domain declined', 'splits': splits, 'manifest': dataset['manifest'],
                                  'full_publication_corpus_audited': False}
    else:
        report['climatecheck'] = {'status': 'not_installed'}
    # Archived materials experiments are optional, not an installation requirement.
    if (data_root / 'matter_of_fact').exists():
        mof, paper_sets = {}, {}
        for split in matter_of_fact.FILES:
            all_cases = matter_of_fact.load(data_root / "matter_of_fact", split=split, scope="all")
            eee = matter_of_fact.load(data_root / "matter_of_fact", split=split, scope="eee")
            paper_sets[split] = {row["source_paper_id"].split("v")[0] for row in all_cases["gold"].values()}
            mof[split] = {"all_cases": len(all_cases["inputs"]), "eee_cases": len(eee["inputs"]),
                "eee_labels": dict(Counter("feasible" if row["label"] else "infeasible"
                                           for row in eee["gold"].values())),
                "eee_version": eee["version"]}
        report['matter_of_fact'] = {'role': 'archived exploratory dataset; outside primary selection',
            'splits': mof, 'paper_overlap_between_splits': {
                f'{a}:{b}': len(paper_sets[a] & paper_sets[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))},
            'manifest': all_cases['manifest']}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    report = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
