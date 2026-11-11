# NLP project summary

Open [the interactive Archify architecture](project-summary.html) and [the full written summary](PROJECT_SUMMARY.md).

The map follows the integrated claim-to-verdict journey, with separate PDF, replay, quality-warning and optional attribution branches. The written summary covers all five contributions, technical dependencies, algorithms, source pins, measured results, tradeoffs and remaining work.

## Explore

- Click a component to focus it and inspect its source evidence.
- Use the Diagram Guide, search, route and lens controls to explore the authored connections.
- Switch light/dark appearance or pause/resume motion with the viewer controls.
- The HTML is self-contained and opens locally. Export options are available in its top-right menu.

Repository evidence is verified against committed `abaaff7cd76749a8d93e75c7d747cb349e1f5f91`. The existing untracked root `docs/`, model weights, environments and runtime caches were excluded. The map embeds verified source paths and line ranges in local-only mode; the written report also links to pinned GitHub sources for authorized repository users. Source code is not uploaded to a hosted diagram service. The summary is an implementation/evidence review, not a new scientific evaluation.

## Reproduce

Archify version 3.2.0 was used from repository commit `7e45f36c66de1254d5ad6ba1d878f6499ea269fd`. Its pinned standalone ZIP was downloaded into the module's ignored `.tmp` build directory. No global skill installation or application dependency was added.

From `explanation_frontend`, using an available Node.js 18+ and that Archify package:

```text
node PATH_TO_ARCHIFY/bin/archify.mjs finalize architecture .archify/architecture-nlp-capstone-20261111-162650/candidate.json .archify/architecture-nlp-capstone-20261111-162650/project-summary.html --repo-root .. --quality showcase --out-dir .archify/architecture-nlp-capstone-20261111-162650/new-review --json
```

Use a fresh review directory for a revised candidate. Generated receipts bind to exact artifact bytes and local paths; regenerate them when rebuilding elsewhere. `.gitattributes` preserves HTML/JSON bytes through Windows checkouts.

Archify is third-party open-source rendering software. Its license, notice and third-party notices are included as `ARCHIFY_*` files. The authored project content is separate from the renderer. Project sources and results remain those of the five-person NLP project.
