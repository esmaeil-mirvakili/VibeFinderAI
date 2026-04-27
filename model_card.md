# Model Card: VibeFinder AI

## Intended Use

VibeFinder AI is a course-demo music recommendation system built over the Kaggle **Audio features and lyrics of Spotify songs** dataset. A user describes the kind of songs they want in English, and the system returns recommendations grounded in the dataset.

It is intended for:

- demonstrating an applied AI workflow with retrieval, agent orchestration, and reliability checks
- exploring music recommendation based on lyrics, metadata, and audio features
- evaluation experiments comparing the full system against ablations

It is not intended for production music recommendation, user profiling, or unrestricted music discovery outside the provided dataset.

## System Behavior

The system implements a LangGraph recommendation workflow with these stages:

1. preference extraction from a natural-language query
2. retrieval planning
3. grounded candidate retrieval
4. deterministic scoring
5. candidate verification
6. critique and optional revision
7. explanation generation
8. reliability reporting

Deterministic tools handle metadata retrieval, audio-feature filtering, FAISS-based Lyric RAG, candidate scoring, and reliability output. LLM-backed stages handle interpretation, planning, critique, and explanations. The system does not use a hand-written lyrical concept taxonomy; lyrical and semantic intent comes from the query interpretation step and is grounded through retrieval and verification.

## Data

The system is limited to the approved columns from the Spotify songs dataset stored in `spotify_songs.csv`. It uses track metadata, playlist metadata, lyrics, language, popularity, and audio features such as energy, danceability, acousticness, valence, and tempo.

The project does not use:

- external music APIs
- scraped metadata
- user listening history
- invented semantic columns outside documented derived artifacts like the FAISS lyric index metadata

## Strengths

- Inspectable intermediate state with structured `rationale`, `issues`, and `warnings`.
- Provider-agnostic LLM interface with Gemini, OpenAI-compatible, Ollama, local, and self-hosted adapters.
- FAISS-based Lyric RAG over embedding vectors from the dataset lyrics.
- Deterministic scoring and reliability checks.
- Ablation variants for critic/revision and Lyric RAG, with multi-step reasoning kept as a default core behavior.
- Streamlit demo app plus separate human-evaluation UI.

## Limitations and Bias

- Recommendation quality depends heavily on the selected LLM and its ability to emit valid structured JSON.
- Local Ollama models may produce weaker preference extraction and retrieval plans than hosted models.
- Automatic metrics cannot fully prove lyrical meaning; the runner reports Lyric RAG and evidence proxy metrics instead.
- The dataset is fixed; the system cannot recommend tracks outside the Kaggle dataset.
- The current UI is a demo interface, not a production recommendation product.
- Dataset coverage bias matters. Underrepresented genres, languages, and lyrical themes in the source data are less likely to be retrieved or ranked well.
- The system may retrieve songs that are semantically related but still miss a subtle perspective constraint, such as betrayal from the cheater's point of view rather than the victim's.

## Reliability and Guardrails

The system includes several guardrail layers:

- structured output schemas for all agent stages
- deterministic retrieval and scoring tools
- verifier and critic stages before final output
- bounded revision instead of open-ended retry loops
- confidence and warning output in the final response
- resumable evaluation and judged comparison scripts

When the system cannot satisfy a request cleanly, it is designed to lower confidence and emit warnings rather than silently present a weak match as certain.

## Evaluation

The repository includes `scripts/run_evaluation.py` and `evaluation/benchmark_queries.json`. The runner executes benchmark queries across the full system and ablations for no critic/revision and no Lyric RAG. It writes resumable per-run query/variant files under `evaluation/results/runs/`, retries failed runs on the next execution, applies a per-run cooldown by default, and writes final JSON/Markdown reports under `evaluation/results/` after all selected runs complete. The current benchmark file is a compact 10-query set focused on lyric-theme, mixed, and hard-constraint queries. These automatic metrics are diagnostics for confidence, warnings, candidate counts, language/genre/subgenre match, full-text metadata match, feature fit, feature-exclusion pass rate, expected retrieval mode recall, Lyric RAG usage, lyric evidence contribution, diversity, and revision use.

The primary quality evaluation is a blinded pairwise human or LLM-as-judge workflow. `scripts/build_judge_tasks.py` turns saved variant outputs into full-vs-ablation tasks, `scripts/run_llm_judge.py` can label those tasks with the configured LLM backend and applies a default cooldown between judge calls, and `scripts/aggregate_judgements.py` now generates both the aggregated judge report and the visualization bundle. It reports full-system win rates, ablation win rates, tie rates, score deltas, failure flags by component and query group, and writes the charts and CSV summaries used for analysis. Human labels use the same judgement schema and aggregation path.

Current repository results show:

- removing Lyric RAG causes the clearest quality drop in both automatic metrics and judged comparisons
- critic/revision helps in some cases, but its effect is smaller and less consistent than Lyric RAG
- weak structured output from smaller or local models can make the workflow fragile even when the rest of the pipeline is grounded

## Future Work

- Add inter-judge agreement reporting when multiple human judges label the same tasks.
- Improve prompt robustness for weaker local LLMs.
- Improve handling of subtle perspective constraints and lyrical point of view.
- Tighten explanation coverage and repair logic for weaker models.
- Extend judged evaluation with more human labels and stronger inter-judge analysis.

## Personal Reflection

This project showed me the difference between a clean scoring simulation and a real multi-stage AI workflow. The original Module 3 recommender was easy to reason about because every score came from a simple formula. Once I moved to LLM-based extraction, retrieval planning, and explanation, the hard part became not only recommendation quality but also workflow stability, traceability, and failure handling.

The biggest surprise in testing was how hard it was to make different LLMs reliably produce valid JSON. The agentic workflow became fragile whenever a model drifted from the required schema, and fixing that required repeated prompt engineering and more system-prompt context than I originally expected. AI assistance was helpful when it suggested embedding the required output schema directly into prompts, which improved structured-output reliability. AI assistance was less helpful when it suggested an overcomplicated project plan with side features like login and user management that did not support the course goals. That reinforced the value of keeping the project focused on grounded retrieval, evaluation, and reliability rather than peripheral product features.
