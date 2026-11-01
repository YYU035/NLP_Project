# Free-only Gemini relevance assessment

Implemented and tested offline on 31 October 2026. **No Gemini or Google
control-plane API calls have been run. No billing or tier settings were changed.**
Retrieval clients, query generation and the search pipeline are unchanged.

## Verified documentation and availability

Google lists **`gemini-2.5-flash`** as a stable model with structured-output
support, but now restricts 2.5 access to users who previously used those models.
New projects are directed to newer models. See the
[official model page](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash).
Therefore, the preferred model is implemented as the default, but access for
your particular API key is **not verified** and must not be assumed.

The [pricing page](https://ai.google.dev/gemini-api/docs/pricing) lists free-tier
standard input/output for 2.5 Flash and `gemini-3.5-flash-lite`. Batch/Flex and
server-side context caching do not have the same free eligibility, so this
evaluator uses ordinary `generateContent` and **local SQLite caching only**.
There are no grounding, browsing, code-execution, file-upload or paid-model calls.

Google's [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits)
directs users to AI Studio for their active model/project quotas. There is no
universal numeric free allowance verified here. Quotas are per project, not per
API key; requests-per-day reset at midnight Pacific time. The local defaults
of **20 attempts per rolling 24 hours** and **15 seconds between starts** are
conservative application caps, **not a promise of Google's free quota**.
Check RPM, TPM and RPD in your AI Studio project before running. If its quota is
zero, do not run; if smaller, lower the local cap/increase the interval.

Both allowed models support [JSON structured output](https://ai.google.dev/gemini-api/docs/structured-output).
The [3.5 Flash-Lite model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
documents that option for new projects. A model is chosen **once at preparation**
and frozen in the manifest. There is no automatic fallback or mid-run switch.
Reported server model versions are checked across responses; a version change
blocks further generation. An unchanged alias cannot guarantee unchanged
underlying weights or bit-for-bit deterministic outputs.

## Why the runner checks billing

An API key does not itself mean “free tier.” The same model has paid pricing.
Following [Google's billing documentation](https://ai.google.dev/gemini-api/docs/billing),
use a project with **no linked billing account**. Do not use trial credits,
prepaid balance, a suspended paid account, Vertex AI, or a billing-enabled key.
Do not add a card, enable billing, prepay, upgrade, or request paid quota.

Before generation, the evaluator:

1. Uses read-only [API-key lookup](https://docs.cloud.google.com/api-keys/docs/reference/rest/v2/keys/lookupKey)
   to identify the actual project owning the key; an unrelated project cannot
   be substituted as evidence of disabled billing.
2. Reads that project's [billing information](https://docs.cloud.google.com/billing/docs/reference/rest/v1/projects/getBillingInfo).
   It requires `billingEnabled: false` and an empty billing-account name.
   It refuses unknown status, linked accounts, permission failures and errors.
3. Checks that the frozen model supports `generateContent` for the key.
4. Rechecks billing immediately before **each** generation request.

Only read-only management endpoints are used; the code contains no billing,
service-enablement or tier-upgrade operations. Keep the project unbilled for
the entire run and do not let another administrator change it concurrently.
If verification cannot succeed, the run stops; there is no bypass flag. This
guard cannot lock Google-side administrative state between the check and POST.
Free access depends on Google's eligibility and continued service availability.

## Setup, without enabling billing

1. In [Google AI Studio](https://aistudio.google.com/apikey), create/select a
   **Free Tier** project with **no billing account linked** and create a key.
   Do not click “Set up billing” or start a paid/free-credit trial. Confirm that
   2.5 Flash is available to that project and note its active quotas.
2. The strict billing guard also needs a short-lived Google OAuth access token
   for the same Google account. With the Google Cloud CLI installed, use
   `gcloud auth login` and then `gcloud auth print-access-token`.
   The account needs `apikeys.keys.lookup` on the key's project and
   `resourcemanager.projects.get` to read billing status (project owners usually
   already have these). The token must have the `cloud-platform` scope, as
   normal Google Cloud CLI user tokens do. No service-account key file is needed.
3. If management API access reports `SERVICE_DISABLED`, the API Keys and Cloud
   Billing management APIs may need enabling **on that same unbilled project**.
   This is API service activation, not linking billing. Do not enable billing
   to resolve an error; stop if your organization does not allow these reads.
   The evaluator never activates services itself.
4. Export secrets locally; `.env` is not automatically read:

```powershell
$env:GEMINI_API_KEY = Read-Host 'Gemini key from your unbilled Free Tier project'
$env:GOOGLE_OAUTH_ACCESS_TOKEN = gcloud auth print-access-token
```

Do not paste keys or tokens into this chat or commit them. OAuth tokens expire;
refresh the token before each later session. The existing project dependencies
(`httpx`, Pydantic and Python's SQLite) suffice; no Google SDK or paid service is
required. Free-tier content may be used by Google to improve its products, as
described in its pricing/terms; this workflow sends only saved public metadata.

## Commands

The default output is `artifacts/gemini-relevance-20261031/`, separate from the
retrieval experiment. **Preparation below has already been run offline**:

```powershell
.venv\Scripts\python.exe scripts/evaluate_relevance.py prepare
```

It verifies the CSV against saved result positions, original claims, paper
identities, titles/abstracts and PDF metadata. The existing conservative identity
matcher gives **250 rows → 234 claim–paper pairs**, including **29 without an
abstract**. One pair has two nonempty abstract variants; the longest version is
selected, with lexical tie-breaking, independent of provider, strategy and rank.
The exact text, source hashes, model, prompt and schema are frozen in the manifest.

After checking your Free Tier project and exporting the credentials, run this
exact command **yourself when ready** (not run during implementation):

```powershell
.venv\Scripts\python.exe scripts/evaluate_relevance.py run --run-live --max-calls 20 --daily-limit 20 --interval 15
```

Use smaller limits if required by AI Studio. Repeat the same command in later
quota windows; valid judgments are reused. At the default local cap, assessing
234 distinct inputs can require roughly 12 sessions spread over at least
12 rolling-day windows, possibly longer with smaller quotas or errors.
Do not rotate projects/keys to evade limits. A 429 stops immediately, without
retrying; `Retry-After`/Google RetryInfo produce a persisted cooldown. Daily
quota violations impose at least 24 hours. Local caps count failed attempts too.
The local ledger is specific to this evaluation; other project usage still
consumes Google's shared quota.

For a **new project lacking 2.5 access**, choose the documented free-tier
alternative before any judging, in a new output directory:

```powershell
.venv\Scripts\python.exe scripts/evaluate_relevance.py prepare --model gemini-3.5-flash-lite --output artifacts/gemini-relevance-flash-lite
.venv\Scripts\python.exe scripts/evaluate_relevance.py run --run-live --output artifacts/gemini-relevance-flash-lite --max-calls 20 --daily-limit 20 --interval 15
```

That is a separate fixed-model evaluation, not a fallback within the 2.5 run.
Availability and zero-price eligibility must be confirmed for that project too.
The future run performs read-only verification plus generation; `prepare` and
`report` are entirely offline and need no credentials:

```powershell
.venv\Scripts\python.exe scripts/evaluate_relevance.py report
```

## Assessment, cache and metrics

The LLM sees **only the original claim, selected title and abstract**. It receives
no source/provider, query, strategy, rank, DOI, URL or retrieval identity. Inputs
are evaluated in fixed opaque-hash order. Labels use one frozen prompt:

- **0:** irrelevant to the substantive subject.
- **1:** topically relevant without clearly useful findings/analysis.
- **2:** potentially useful evidence for assessing the claim, including
  contradictory evidence. This is not a claim-support or truth label.

Strict local Pydantic validation checks integer labels, justification length and
an exact supplied-text quote; label 2 requires a quote. Safety blocks, truncated
responses and malformed JSON never become label 0. Missing abstracts are flagged
and assessed using the title only, with no invented findings.

Cache keys include the full input, model, prompt, schema and generation settings.
Reservations are committed before sending, so successful items and ambiguous
timeouts/interrupted requests are not automatically resent. An `uncertain`,
`inflight` or `error` item stays unresolved; review it explicitly rather than
deleting the cache and blindly resending. This favors no duplicate calls over
automatic completion when the network outcome is unknowable. Explicit rejected
429s are safe to resume after cooldown. A lock prevents concurrent CLI runs;
after a crash, inspect the PID in `.run.lock` and remove that file only after
confirming the old process stopped. Keep the cache and frozen manifest together.

Outputs:

- `cache.sqlite3`: durable judgments, attempt counts, cooldowns and versions.
- `judgments.csv`: pair IDs, labels, explanations, quotes and assessment status.
- `metrics.json` / `metrics.csv`: P@10, relevant pair yield and relevant PDF URL
  availability at thresholds **label ≥1** and **label =2**.
- `displacement.csv`: relevant concept-only additions, original-only losses,
  overlap, net change and the two-query union yield.
- `report.md`: readable progress and comparison summary. **No relevance metrics
  have been measured yet**; incomplete metrics remain null/pending.
- `manual_validation_30.csv`: **30 fixed examples, three per claim**, hiding
  retrieval provenance and model predictions. Human-label fields are blank.

P@10 counts relevant unique identities in each provider's ten result slots;
valid empty searches score zero, while provider failures are excluded and
counted. Aggregate P@10 macro-averages over claims. No combined-provider P@10
is invented without a merged ranked top ten. Yield and displacement use unique
claim–paper pairs and deduplicate across providers for the combined view.
Relevant PDF fractions use URL metadata from the evaluated arm, not another
arm's enriched record. URLs are not downloaded or verified.

Compare equal-budget original/concept arms first (ten requests per arm/provider
in this experiment). The original-plus-concept union costs twice as many
requests and is reported separately. LLM-estimated relevance is not ground truth;
there is no corpus recall or claim-accuracy estimate. For optional validation,
two humans can independently label the 30 blinded examples, adjudicate differences,
then check agreement, weighted kappa and false label-2 judgments. Thirty examples
give a preliminary reliability check, not a precise guarantee across disciplines.

Validation: **216 pytest tests passed** in 7.76 seconds; **Ruff passed**.
Twenty new mocked/offline tests cover blinding, strict labels, quoted evidence,
deduplication, rate-limit cooldowns, durable quota reservations, timeout resume,
billing refusal/rechecks, model-version changes, metrics and CLI opt-in. The
actual offline preparation has **0/234 judgments** and 30 blank manual examples.
