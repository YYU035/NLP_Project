# Consolidation validation — 11 November 2026

The release reorganizes the integrated project from `kireeti` snapshot `11ba8dc` into function-based packages. This is a software migration, not a new accuracy study.

| Check | Result |
|---|---|
| Independent Python environment | Pass: Python 3.12.15, no system packages; local source imports verified |
| Software tests | 416 passed; three upstream Streamlit/Pandas deprecation warnings |
| Dependency compatibility | `pip check` passed |
| CLI and Windows entry points | Help, wheel build, PowerShell parse and invocation from another working directory passed |
| App server | Health and page both HTTP 200 on loopback |
| Original versus renamed real-model route | Same claim, paper order, exact evidence, verification, aggregation and explanation content |
| Ranking numerical comparison | Maximum BM25 score difference 8.89e-16; comparison tolerance 1e-12 |
| Cached selection changes | 20 changes, zero extra neural passage calls |
| Constructed one-page PDF | Real local inference, page references, grounded explanation and replay passed |
| Fresh public discovery | Nine candidates, three verified papers/three passages, grounded SUPPORTED result and replay passed |
| Migration integrity | 435 entries checked: 312 preserved byte-for-byte, 123 transformed with recorded hashes |
| Frozen study evidence | Four manifests, 179 artifact checks passed |
| Demo assets | 21 benchmark assets, eight replays and one recorded attribution checked |
| Dataset staging | Official SciFact archive and all three file hashes verified; cached reuse and hostile archive cases tested |

The real-model checks reused existing pinned model files through explicit, ignored local model paths. They did not download a second copy of the weights. The fresh environment reused only third-party package download caches, not the old virtual environment. The live query is a functional check, not an independently adjudicated accuracy example.

Run-specific timestamps, timings, fingerprints/context keys and hashes containing timing metadata differ across executions. Verification/aggregation objects and explanation content matched exactly; ranking scores were compared with the stated tolerance. Models, thresholds, schemas and primary/baseline policies retain their earlier scientific meaning. No policy was tuned on the inspected audit.

The original frozen reports and examples retain their evidence bytes. New experiment publication goes to ignored outputs rather than replacing the recorded study bundles. Historical owner-checkout QC tools remain as source records under `docs/history/reproduction_tools/`; current setup, application and dataset staging do not need those old branch objects.

See [the machine-readable receipt](migration_validation.json), [the migration path/hash manifest](../architecture/migration_manifest.json), [existing scientific results](index.md), and [current architecture](../architecture/overview.md). Existing scientific limitations—contradiction recall, rationale coverage, failed IG gates, rejected local generation and pending human reviews—remain open.
