# PDF evidence review — AI-assisted draft

Status: all 12 passages inspected against eight original page images. Human sign-off remains pending. These judgments are not human gold labels and do not replace the SciFact evaluation.

| Case / page | Relevant | Individually sufficient | Clean context | Finding |
|---|---|---|---|---|
| t-architecture / 10 | yes | yes | yes | Conclusion states attention-only architecture and replacement of recurrent layers; surrounding comparison is retained. |
| t-architecture / 2 | yes | no | yes | Relevant to recurrence and attention, but convolution exclusion is not explicit in this selected span. |
| t-base-schedule / 7 | yes | yes | yes | Contains base-model steps, twelve hours and eight P100 GPUs; distinguishes big-model schedule. |
| t-base-schedule / 2 | yes | no | yes | Contains twelve hours and eight P100s but not base-model identity or 100,000 steps. |
| t-big-results / 8 | yes | no | yes | Contains big-model EN-DE 28.4 BLEU but omits the following sentence giving 3.5 days and eight P100 GPUs. |
| t-big-results / 8 | yes | no | no | Flattened table loses column geometry; training duration and GPU count are absent. Consult original table and prose. |
| b-bidirectional / 13 | yes | yes | no | Caption supports bidirectionality in all layers; selected text appends an unrelated column fragment, to converge. |
| b-bidirectional / 4 | yes | no | yes | Pretraining discussion is relevant but does not explicitly establish left and right context in all layers. |
| b-masking-qualification / 16 | yes | no | yes | Related masking ablation introduction, but selected span lacks the 15% population and 80% conditional replacement rate. |
| b-masking-qualification / 4 | yes | yes | yes | Explicitly retains 15% selected positions and conditional 80/10/10 replacement; qualification preserved. |
| b-hardware / 13 | yes | yes | yes | Names BERT Large, sixteen Cloud TPUs, sixty-four chips and four-day pretraining; distinguishes Base. |
| b-hardware / 5 | no | no | no | SQuAD fine-tuning footnote is irrelevant to BERT Large pretraining hardware; passage crosses prose and footnote boundary. |

All 12 source spans are locatable. On this AI-reviewed convenience sample, 11/12 are topically relevant and 5/12 are individually sufficient; these are descriptive counts, not benchmark performance estimates.

Priority findings: preserve the following hardware/schedule sentence for Transformer big results; avoid flattened-table evidence without its layout; remove caption-to-column fragments and unrelated footnotes; prefer the actual masking-rule paragraph over its ablation introduction. Do not alter frozen benchmark settings using this sample without reporting a separate development iteration.

The original human-review.csv remains blank. A human can accept or correct each draft judgment using the original packet; no teammate confirmation has been fabricated.
