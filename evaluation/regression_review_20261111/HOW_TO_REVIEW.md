# How to review the nine cases

Open index.html first. For each scientific claim, compare the candidate verdict with the exact selected evidence, then open the original abstract. Yellow sentences are benchmark rationales, not newly generated explanations. Expand both passage lists to see complete model input; the counterfactual column uses old recorded predictions with the new retained paper scope, without any new NLI.

Use yes / no / uncertain / not_applicable in the blank REVIEW_FORM.csv:

- evidence_supports_verdict: does the selected evidence justify the candidate's displayed verdict in this selected-source scope? Do not mark yes just because the benchmark label matches. For no-positive benchmark cases, the supplied annotations may be incomplete; record your source-based reasoning.
- citation_correct: do the quoted text, document identity and (for PDF) page match the source shown?
- context_helpful: compared with focused sentences, does the extra context make the evidence easier to interpret? Use not_applicable for the unchanged PDF policy.
- issue_notes: note the specific sentence, missing qualification, population, outcome or source mismatch that explains your judgment. Use uncertain when specialist knowledge is needed.
- reviewer_name: initials or a chosen reviewer label are sufficient.

For the PDF, open the original linked paper and import the saved replay in the app. Check the recorded pages, reading order and quoted text. Its scientific verdict has no independently adjudicated reference label.

Save your completed copy under explanation_frontend/outputs/regression_review_20261111/your-name-review.csv so the blank template stays unchanged. No ratings have been completed automatically. The generator refuses to overwrite a populated template.

This is a deliberately selected diagnostic review, not a blind user study or an estimate of human agreement. Do not turn these selected examples into a new model accuracy score. The fixed 209-claim audit and all measured outcomes remain unchanged. Any policy experiments must use separate development data.
