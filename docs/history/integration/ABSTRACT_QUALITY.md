# Advisory abstract-quality checks

Raghavi's `abstract-quality-v1` rules from commit `0ee9565` are now available in the frontend. They flag missing descriptions, possible cut-offs, unusually long/body-like text, placeholders and uncertain text structure. These are review heuristics, with no measured classification accuracy. **No warning detected does not mean complete, relevant or scientifically correct.**

## Use in the app

1. Open **Find papers** and load recorded, uploaded or live discovery results. Candidate cards show an abstract-check label. Inspect a paper to see its exact stored text and warning reasons.
2. Expand **About the abstract checks and download** for the separate JSON report. Candidate counts refer to every stored candidate, independently of the view filters. Provider occurrences are a separate report scope and must not be added to the candidate count.
3. Source-backed results in **Verify evidence → Papers & passages** retain the check labels and expose the stored provider abstract. **Run details** offers the report. Replays receive current advisory checks on their stored source text; saved model predictions remain unchanged.

PDF passages and synthetic demo fragments are not assessed as provider abstracts. If a legacy handoff omits stored provider records, occurrence counts explicitly cover only available records. If stored occurrences cannot be linked to retained candidates, that scope is withheld. Candidate checks remain available in both cases, and the report marks provider-occurrence coverage incomplete.

Checks run offline using existing dependencies. They never rewrite source text, deselect papers, alter ranking, change model inputs or produce a scientific verdict. Discovery exports and replay schemas/fingerprints are unchanged. The report retains exact text hashes, source IDs, aliases and provenance.

## Optional saved-result batch report

The existing pinned discovery package is needed only for this CLI's native saved-result models and prefix merge. Point it at a directory containing native `cNN.json` archives:

```powershell
explanation_frontend\.venv\Scripts\python.exe -m explanation_frontend._upstream.discovery_quality.assess_saved --input PATH_TO_SAVED_SEARCHES --output explanation_frontend\outputs\abstract-quality-review --depths 10 30
```

The output directory must be new and separate from the inputs. The command writes JSON, CSV and Markdown without API calls or edits to source archives. The isolated copy fixes the upstream empty-result CSV bug: zero-result batches now produce all three files, including CSV headers. That fix does not alter the classification rules.

The installed live-search package remains pinned at `259d2a9`; this helper is independently pinned inside `_upstream/discovery_quality/`. [SOURCE.json](_upstream/discovery_quality/SOURCE.json) records original/adapted file hashes and the exact import/reporting adjustments. No teammate directory was merged or edited.
