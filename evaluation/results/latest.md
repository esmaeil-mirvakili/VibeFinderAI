# VibeFinder Evaluation Summary

- Generated at: 2026-04-24T23:05:35.265575+00:00
- Query count: 10
- Variants: full, no_critic_revision, no_lyric_retriever
- Model backend: gemini
- Model name: gemini-3.1-flash-lite-preview

| Variant | Runs | Success | Avg Confidence | Avg Constraint | Avg Feature Fit | Text Match | Mode Recall | Lyric Evidence | Avg Warnings | Revisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 10 | 10 | 0.496 | 0.870 | 0.610 | - | 0.967 | 1.000 | 14.300 | 0 |
| no_critic_revision | 10 | 10 | 0.498 | 0.858 | 0.610 | - | 0.917 | 1.000 | 11.900 | 0 |
| no_lyric_retriever | 10 | 10 | 0.389 | 0.388 | 0.629 | - | 0.433 | 0.000 | 19.000 | 7 |
