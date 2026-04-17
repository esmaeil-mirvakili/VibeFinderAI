# VibeFinderAI

VibeFinder AI is an agentic music recommendation system that helps users find songs by describing what they want in plain English. Instead of relying only on simple filters like genre or artist, the system interprets the user’s request to identify deeper preferences such as lyrical themes, emotional tone, perspective, energy level, and style. For example, a user can ask for songs about betrayal, regret, confidence, or heartbreak, along with sonic preferences like high energy, acoustic sound, or a specific genre.

The system works through a multi-step workflow. It first extracts structured preferences from the user’s query, then decides the best retrieval strategy for that request. Depending on the query, it may rely more heavily on lyric-based retrieval, metadata filters, audio-feature filtering, or a hybrid of all three. After retrieving candidate songs, the system verifies whether they truly match the request, critiques weak or narrow result sets, and revises its search when needed. Finally, it ranks the best songs, generates grounded explanations for why each song was selected, and outputs confidence levels and warnings when the match is weak or uncertain.

VibeFinder AI is designed to be both explainable and reliable. Each recommendation is supported by evidence from the dataset, such as lyrics, genre, album details, language, and audio features like energy, danceability, acousticness, and tempo. The system also includes verification, self-critique, and revision steps so it can catch weak results instead of returning them blindly. The final goal of the project is to show how retrieval, agentic reasoning, and reliability checks can be combined into a complete applied AI system for music recommendation.

## Architecture Overview

The planned system follows the staged workflow in `assets/architecture.mmd` and `assets/flowchart.mmd`:

User query -> preference extraction -> retrieval strategy -> lyric/metadata/feature retrieval -> critique and optional revision -> ranking -> explanations -> confidence and warnings.

The current codebase includes the dataset loader, Streamlit demo shell, shared tool runtime, and deterministic retrieval tools for metadata, audio features, and lyrics. Agent extraction, scoring, ranking, reliability, and evaluation are still pending.

Tool implementation should follow `agent_tools.md`, which defines the deterministic retrieval, filtering, scoring, reliability, and tool-registry plan that agents will call through LangGraph.

## Setup Instructions

python requirement: 3.11

Implementation setup instructions will be added once the pipeline modules are built.

## Sample Interactions

Example target queries for the final LLM-backed system:

- `English pop songs about betrayal with high energy and low acousticness`
- `Sad English songs about regret after cheating`
- `Spanish dance songs with positive valence and high tempo`

Sample outputs will be added after the working pipeline exists.

## Design Decisions

The Python package keeps the agent-facing types and LLM interface shape small while adding deterministic retrieval tools in scoped modules under `src/vibefinder/tools/`. New implementation logic should continue to follow `design.md`, `agent_tools.md`, and `agent_design.md`.

The lyric RAG layer uses an in-memory FAISS index over real embedding-model vectors generated from the dataset `lyrics` column. The same embedding model is used for lyric indexing and query embedding through a provider-agnostic `TextEmbeddingProvider` interface. The default provider is SentenceTransformers with `sentence-transformers/all-MiniLM-L6-v2`, and it can be changed with `VIBEFINDER_EMBEDDING_MODEL`.

On app startup, VibeFinder loads the lyric FAISS index from the project root if it already exists. If it does not exist, the app builds it from `spotify_songs.csv` and stores these generated files in the project root:

- `lyric_faiss.index`
- `lyric_faiss_metadata.json`

## Testing Summary

The current tests cover the metadata retrieval tool, feature filter tool, lyric retrieval tool, registry exposure, prompt-safe schemas, and `ToolRunner` success/error paths.

## Reflection

TODO: Fill this in after the full implementation and evaluation are complete.
