# Free/open-source operation — 4 November 2026

The proprietary hosted explanation client has been removed. No subscription, paid inference API, API key or hosted model endpoint is used by this frontend's explanation workflow.

## Explanation and evaluation

- Templates, citation checking, replay and human-review forms run locally with no external request.
- Optional generation uses Qwen/Qwen2.5-0.5B-Instruct, Apache-2.0, at revision 7ae557604adf67be50417f59c2c2f167def9a775, through open-source PyTorch/Transformers on CPU.
- Inference uses local_files_only=True and trust_remote_code=False. It cannot download a missing model or fall back to a hosted service. The evaluation CLI's --mode llm uses the same local client.
- setup-local-llm.ps1 explicitly downloads public model/tokenizer/license files once from Hugging Face, without an account token. Hugging Face hosted inference is not used. Weights and caches stay in this module's ignored directories.
- Missing weights, generation failures or invalid/unsupported output leave the template available. A small local model's scientific usefulness is an evaluation question, not an assumed quality improvement.

## Remaining public discovery endpoints

| Endpoint | Purpose | Credential/payment policy |
| --- | --- | --- |
| https://api.openalex.org/works | Public paper metadata | Anonymous basic queries only; credentials are cleared. OpenAlex offers paid higher-volume access, but this app cannot use that account configuration. |
| https://api.openaire.eu/graph/v3/research-products | Public OpenAIRE research metadata | The pinned upstream client's anonymous path; credentials are cleared. |
| https://export.arxiv.org/api/query | Public arXiv metadata | No API account or payment credentials. |

Only these three discovery providers are enabled by this frontend factory. The installed upstream package's optional evaluators and other providers are not imported or called by this workflow. Scientific claim text is sent to these public catalogues only when the user requests Live search; explanation text/evidence is never sent to hosted inference.

Provider pacing and rate limits remain in force. Quota, availability or authentication failures are reported as partial/unavailable search; there is no paid upgrade, credit purchase, payment fallback or credential forwarding. Catalogues are external services with their own operating terms; anonymous access can become unavailable. Cached and template modes remain usable offline.

Links to paper landing pages/PDF metadata are inspectable links. This frontend does not fetch arbitrary paper URLs or execute their contents.

## Sources checked

- [Qwen model and Apache-2.0 license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [OpenAlex free anonymous basic API access](https://help.openalex.org/api/authentication/)
- [OpenAlex open-source code statement](https://explore.openalex.org/about)
- [OpenAIRE public Graph API](https://graph.openaire.eu/docs/apis/graph-api/)
- [arXiv API access](https://info.arxiv.org/help/api/index.html)

The discovery package is pinned to teammate source revision 259d2a9. No teammate source file was edited to enforce this frontend policy.

## Integrated local retrieval and NLI - 6 November 2026

The built-in pipeline now runs local Naila BM25/SBERT and Yu Yue RRF, exact abstract-span selection, and Chong Yao NLI verification. NLI uses the MIT-licensed model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli at b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7. setup-pipeline.ps1/pipeline_setup.py explicitly download its public model files once; the weight checksum is verified before activation. Tokenizers/models load with local_files_only=True and trust_remote_code=False at runtime. The upstream constructor that can download revision main is not invoked. Anonymous Hugging Face file download is setup-only, not hosted inference.

Verify evidence -> Live explicitly requests public discovery, so its claim is sent to the same three catalogues listed above. All downstream processing remains on this computer. Recorded discovery -> local verification, cached subsets, templates and replay require no external requests. No paid/proprietary API or additional credentials were added.

## QC model checks - 6 November 2026

The Qwen setup/download is now complete; 12 actual local generations were checked and all failed the strict schema, with safe template fallbacks. No hosted generation was used. Optional evaluation/reranker_qc.py explicitly downloads public Apache-2.0 cross-encoder/ms-marco-MiniLM-L6-v2 files at 233902d25c440f23af6f7d6e94d2946bac0bee0a if absent, checks the weight hash and runs locally. This is a QC tool, not an automatically enabled product dependency. Source-test QC blocks external network connections; loopback is permitted for Windows asyncio. No paid/proprietary judging or inference API was called.
