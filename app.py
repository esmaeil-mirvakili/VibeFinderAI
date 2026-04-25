"""Streamlit chat UI for VibeFinder AI."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
ENV_FILE = PROJECT_ROOT / ".env"
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def load_project_env(path: Path = ENV_FILE) -> int:
    """Load simple KEY=VALUE entries from the project .env without overriding shell env."""

    if not path.exists():
        return 0

    loaded_count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned_value = value.strip().strip('"').strip("'")
        os.environ[key] = cleaned_value
        loaded_count += 1
    return loaded_count


def configure_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    if os.getenv("VIBEFINDER_FILE_LOGGING_CONFIGURED") == "1":
        return
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message} | {extra}"
    logger.add(
        sys.stderr,
        level="INFO",
        format=log_format,
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        LOG_FILE,
        level="INFO",
        format=log_format,
        rotation="1 MB",
        retention=5,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    os.environ["VIBEFINDER_FILE_LOGGING_CONFIGURED"] = "1"


configure_logging()
loaded_env_count = load_project_env()
logger.info("project_env_loaded", path=str(ENV_FILE), loaded_count=loaded_env_count)

from vibefinder.config import load_retrieval_prompt_config
from vibefinder.data import load_songs_dataset
from vibefinder.embeddings import get_default_embedding_provider
from vibefinder.graph import run_recommendation
from vibefinder.graph_runtime import create_graph_runtime_context
from vibefinder.tools import ensure_lyric_index
from vibefinder.variants import VariantConfig

CHAT_PLACEHOLDER = "Describe the kind of music you are looking for."
MAX_UI_RECOMMENDATIONS = 5


@st.cache_resource(show_spinner="Checking dataset and lyric index...")
def load_app_resources():
    logger.info("dataset_startup_check", dataset_path=".")
    songs = load_songs_dataset(path=".")
    logger.info("dataset_loaded", row_count=len(songs), column_count=len(songs.columns))

    retrieval_prompt_config = load_retrieval_prompt_config()
    logger.info(
        "retrieval_prompt_config_loaded",
        config_sections=list(retrieval_prompt_config),
    )

    embedder = get_default_embedding_provider()
    logger.info("lyric_index_startup_check", root=str(PROJECT_ROOT), embedding_model=embedder.name)
    lyric_index = ensure_lyric_index(
        songs=songs,
        root=PROJECT_ROOT,
        embedder=embedder,
    )
    logger.info(
        "lyric_index_ready",
        indexed_count=len(lyric_index.track_ids),
        dimension=lyric_index.dimension,
        embedding_model=lyric_index.embedding_model,
    )
    return songs, retrieval_prompt_config, embedder, lyric_index


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vf-primary: #6750a4;
            --vf-primary-container: #eaddff;
            --vf-on-primary-container: #21005d;
            --vf-secondary: #625b71;
            --vf-surface: #fffbff;
            --vf-surface-container: #f3edf7;
            --vf-surface-container-high: #ece6f0;
            --vf-outline: #79747e;
            --vf-outline-variant: #cac4d0;
            --vf-on-surface: #1d1b20;
            --vf-on-surface-variant: #49454f;
            --vf-success: #006e1c;
            --vf-shadow: 0 2px 6px rgba(29, 27, 32, 0.08), 0 1px 3px rgba(29, 27, 32, 0.12);
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        footer {
            display: none;
        }

        html,
        body,
        .stApp {
            background: var(--vf-surface);
            color: var(--vf-on-surface);
            font-family: Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1240px;
            padding: 28px 28px 18px;
        }

        .vf-main-row {
            display: grid;
            grid-template-columns: 208px 1fr;
            gap: 12px;
            margin-bottom: 34px;
        }

        .vf-component-panel,
        .vf-chat-panel-frame {
            border: 1px solid var(--vf-outline-variant);
            border-radius: 8px;
            background: var(--vf-surface-container);
            height: 590px;
            box-shadow: var(--vf-shadow);
        }

        .vf-component-panel {
            padding: 14px 6px 12px;
        }

        .st-key-component_panel {
            border-color: var(--vf-outline-variant) !important;
            border-radius: 8px !important;
            background: var(--vf-surface-container) !important;
            box-shadow: var(--vf-shadow);
            padding: 14px 8px 12px !important;
        }

        .vf-component-panel-title {
            color: var(--vf-on-surface);
            font-size: 14px;
            font-weight: 500;
            margin: 0 0 12px;
        }

        .vf-component-label {
            color: var(--vf-on-surface);
            font-size: 13px;
            font-weight: 500;
            line-height: 18px;
            letter-spacing: 0;
            white-space: normal;
            overflow-wrap: anywhere;
            padding: 4px 0;
        }

        .st-key-component_panel * {
            color: var(--vf-on-surface);
        }

        div[data-testid="stToggle"] {
            margin-bottom: 8px;
        }

        div[data-testid="stToggle"] label {
            gap: 10px;
            min-height: 28px;
            align-items: center;
        }

        .st-key-component_panel div[data-testid="stToggle"] p,
        .st-key-component_panel div[data-testid="stToggle"] span {
            color: var(--vf-on-surface) !important;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0;
        }

        .st-key-component_panel div[data-testid="stToggle"] [data-baseweb="switch"] {
            box-shadow: none;
        }

        .st-key-component_panel div[data-testid="stToggle"] [data-baseweb="switch"] div {
            border-color: var(--vf-outline) !important;
        }

        .st-key-component_panel div[data-testid="stToggle"] [aria-checked="true"] {
            background: var(--vf-primary) !important;
            border-color: var(--vf-primary) !important;
        }

        .st-key-component_panel div[data-testid="stToggle"] [aria-checked="false"] {
            background: var(--vf-surface) !important;
            border-color: var(--vf-outline) !important;
        }

        .vf-chat-panel-frame {
            overflow: hidden;
        }

        .vf-chat-panel-frame iframe {
            display: block;
        }

        .vf-chat-panel {
            height: 100%;
            padding: 22px;
            overflow-y: auto;
            box-sizing: border-box;
        }

        .vf-message {
            max-width: 66%;
            padding: 12px 14px;
            border: 1px solid var(--vf-outline-variant);
            border-radius: 8px;
            color: var(--vf-on-surface);
            background: var(--vf-surface);
            margin-bottom: 12px;
            font-size: 15px;
            line-height: 1.45;
            white-space: pre-wrap;
        }

        .vf-message-user {
            margin-left: auto;
            margin-right: 2px;
            background: var(--vf-primary-container);
            color: var(--vf-on-primary-container);
        }

        .vf-result-card {
            width: min(720px, 78%);
            margin: 32px 0 0 18px;
            border: 1px solid var(--vf-outline-variant);
            border-radius: 8px;
            background: var(--vf-surface);
            padding: 18px 24px;
            color: var(--vf-on-surface);
            box-sizing: border-box;
            box-shadow: var(--vf-shadow);
        }

        .vf-result-title,
        .vf-warnings-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .vf-track {
            margin: 12px 0 16px;
        }

        .vf-track-heading {
            font-weight: 600;
            margin-bottom: 4px;
        }

        .vf-track-confidence {
            margin-bottom: 8px;
        }

        .vf-spotify-iframe {
            border: 0;
            border-radius: 8px;
            width: min(370px, 100%);
            height: 80px;
            display: block;
        }

        .vf-why {
            margin-top: 8px;
            line-height: 1.45;
        }

        .vf-warnings {
            margin-top: 14px;
        }

        .vf-warnings ul {
            margin: 8px 0 0 18px;
            padding: 0;
        }

        div[data-testid="stForm"] {
            border: 1px solid var(--vf-outline-variant);
            border-radius: 8px;
            background: var(--vf-surface-container);
            padding: 8px;
            box-shadow: var(--vf-shadow);
        }

        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            gap: 0;
            align-items: center;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] {
            padding: 0;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] > label {
            display: none;
        }

        div[data-testid="stForm"] input {
            border: 0;
            box-shadow: none !important;
            background: var(--vf-surface) !important;
            color: var(--vf-on-surface);
            font-size: 18px;
            font-weight: 400;
            padding: 15px 14px;
            caret-color: var(--vf-primary);
        }

        div[data-testid="stForm"] input::placeholder {
            color: var(--vf-primary);
            opacity: 0.72;
        }

        div[data-testid="stForm"] input:focus {
            border: 0;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] [data-baseweb="input"],
        div[data-testid="stForm"] [data-testid="stTextInputRootElement"] {
            background: var(--vf-surface) !important;
            border: 0 !important;
            box-shadow: none !important;
            border-radius: 8px !important;
        }

        div[data-testid="stForm"] [data-baseweb="base-input"],
        div[data-testid="stForm"] [data-baseweb="input"] > div,
        div[data-testid="stForm"] [data-testid="stTextInputRootElement"] > div {
            background: var(--vf-surface) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] button {
            border: 0;
            border-radius: 8px;
            box-shadow: none;
            background: var(--vf-primary-container);
            color: var(--vf-primary);
            font-size: 28px;
            min-height: 52px;
            transition: background 120ms ease, color 120ms ease;
        }

        div[data-testid="stForm"] button:hover,
        div[data-testid="stForm"] button:focus {
            border: 0;
            box-shadow: none;
            background: #d0bcff;
            color: var(--vf-on-primary-container);
        }

        div[data-testid="stForm"] button:disabled,
        div[data-testid="stForm"] input:disabled {
            color: var(--vf-on-surface-variant) !important;
            opacity: 0.72;
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 16px 12px;
            }

            .vf-main-row {
                grid-template-columns: 1fr;
            }

            .vf-component-panel {
                height: auto;
            }

            .vf-chat-panel-frame {
                height: 520px;
            }

            .vf-message,
            .vf-result-card {
                max-width: 100%;
                width: 100%;
                margin-left: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chat_panel(is_searching: bool = False) -> None:
    messages = st.session_state.get("messages", [])
    message_html = []
    for index, message in enumerate(messages):
        role = message["role"]
        is_latest_message = index == len(messages) - 1
        if role == "user":
            content = html.escape(str(message["content"]))
            target_attr = ' data-scroll-target="true"' if is_latest_message else ""
            message_html.append(f'<div{target_attr} class="vf-message vf-message-user">{content}</div>')
        else:
            rendered_message = _assistant_message_html(message.get("content"))
            if is_latest_message:
                rendered_message = _mark_scroll_target(rendered_message)
            message_html.append(rendered_message)

    if is_searching:
        target_attr = ' data-scroll-target="true"' if not messages else ""
        message_html.append(
            f'<div{target_attr} class="vf-searching">'
            '<span class="vf-spinner"></span>'
            '<span>Searching...</span>'
            "</div>"
        )

    panel_html = f"""
    <!doctype html>
    <html>
    <head>
      <style>
        html, body {{
          margin: 0;
          padding: 0;
          background: transparent;
          font-family: Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .vf-chat-panel {{
          height: 590px;
          padding: 14px 18px 22px;
          overflow-y: auto;
          box-sizing: border-box;
          border: 1px solid #cac4d0;
          border-radius: 8px;
          background: #f3edf7;
          color: #1d1b20;
        }}
        .vf-chat-panel::-webkit-scrollbar {{
          width: 10px;
        }}
        .vf-chat-panel::-webkit-scrollbar-track {{
          background: transparent;
        }}
        .vf-chat-panel::-webkit-scrollbar-thumb {{
          background: #cac4d0;
          border-radius: 8px;
          border: 2px solid #f3edf7;
        }}
        .vf-message {{
          max-width: 66%;
          padding: 14px 24px;
          border: 1px solid #cac4d0;
          border-radius: 8px;
          background: #fffbff;
          margin-bottom: 12px;
          font-size: 15px;
          line-height: 1.45;
          white-space: pre-wrap;
          box-sizing: border-box;
          box-shadow: 0 1px 2px rgba(29, 27, 32, 0.08);
        }}
        .vf-message-user {{
          margin: 0 6px 32px auto;
          background: #eaddff;
          border-color: #d0bcff;
          color: #21005d;
          min-height: 72px;
          display: flex;
          align-items: center;
        }}
        .vf-searching {{
          display: flex;
          align-items: center;
          gap: 12px;
          margin: 18px 0 0 320px;
          font-size: 22px;
          color: #49454f;
          font-weight: 500;
        }}
        .vf-spinner {{
          width: 32px;
          height: 32px;
          border: 4px solid #e7e0ec;
          border-top-color: #6750a4;
          border-radius: 50%;
          display: inline-block;
          animation: vf-spin 0.9s linear infinite;
        }}
        @keyframes vf-spin {{
          to {{ transform: rotate(360deg); }}
        }}
        .vf-result-card {{
          width: min(742px, 82%);
          margin: 32px 0 0 0;
          border: 1px solid #cac4d0;
          border-radius: 8px;
          background: #fffbff;
          padding: 18px 24px;
          color: #1d1b20;
          box-sizing: border-box;
          box-shadow: 0 2px 6px rgba(29, 27, 32, 0.08), 0 1px 3px rgba(29, 27, 32, 0.12);
        }}
        .vf-result-title,
        .vf-warnings-title {{
          font-weight: 600;
          margin-bottom: 8px;
          color: #21005d;
        }}
        .vf-track {{
          margin: 12px 0 16px;
        }}
        .vf-track-heading {{
          font-weight: 600;
          margin-bottom: 4px;
          color: #1d1b20;
        }}
        .vf-track-confidence {{
          margin-bottom: 8px;
          color: #49454f;
        }}
        .vf-spotify-iframe {{
          border: 0;
          border-radius: 8px;
          width: min(370px, 100%);
          height: 80px;
          display: block;
        }}
        .vf-why {{
          margin-top: 8px;
          line-height: 1.45;
          color: #1d1b20;
        }}
        .vf-warnings {{
          margin-top: 14px;
          color: #49454f;
        }}
        .vf-warnings ul {{
          margin: 8px 0 0 18px;
          padding: 0;
        }}
        @media (max-width: 760px) {{
          .vf-message,
          .vf-result-card {{
            max-width: 100%;
            width: 100%;
          }}
          .vf-searching {{
            margin-left: 18px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="vf-chat-panel">{"".join(message_html)}</div>
      <script>
        function scrollToLatestMessage() {{
          const panel = document.querySelector(".vf-chat-panel");
          const target = document.querySelector("[data-scroll-target='true']");
          if (!panel || !target) {{
            return;
          }}
          const top = Math.max(target.offsetTop - 14, 0);
          panel.scrollTo({{ top: top, behavior: "auto" }});
        }}
        requestAnimationFrame(scrollToLatestMessage);
        window.addEventListener("load", scrollToLatestMessage);
        window.setTimeout(scrollToLatestMessage, 150);
      </script>
    </body>
    </html>
    """

    components.html(panel_html, height=590, scrolling=False)


def render_variant_controls() -> VariantConfig:
    with st.container(height=590, border=True, key="component_panel"):
        st.markdown('<div class="vf-component-panel-title">System Components:</div>', unsafe_allow_html=True)
        use_multi_step_reasoning = _component_toggle("Multi-step Reasoning", value=True, key="use_multi_step_reasoning")
        use_lyric_retriever = _component_toggle("Lyric RAG", value=True, key="use_lyric_retriever")
        use_critic_revision = _component_toggle("Critic and Revision Agents", value=True, key="use_critic_revision")

    disabled = []
    if not use_multi_step_reasoning:
        disabled.append("no_multi_step")
    if not use_critic_revision:
        disabled.append("no_critic_revision")
    if not use_lyric_retriever:
        disabled.append("no_lyric")

    return VariantConfig(
        name="full" if not disabled else "+".join(disabled),
        use_multi_step_reasoning=use_multi_step_reasoning,
        use_critic_revision=use_critic_revision,
        use_lyric_retriever=use_lyric_retriever,
    )


def _component_toggle(label: str, *, value: bool, key: str) -> bool:
    switch_col, label_col = st.columns([0.26, 0.74], gap="small", vertical_alignment="center")
    with switch_col:
        selected = st.toggle(label, value=value, key=key, label_visibility="collapsed")
    with label_col:
        st.markdown(f'<div class="vf-component-label">{html.escape(label)}</div>', unsafe_allow_html=True)
    return selected


def _variant_to_session_dict(variant_config: VariantConfig) -> dict[str, Any]:
    return {
        "name": variant_config.name,
        "use_multi_step_reasoning": variant_config.use_multi_step_reasoning,
        "use_critic_revision": variant_config.use_critic_revision,
        "use_lyric_retriever": variant_config.use_lyric_retriever,
    }


def _variant_from_session_dict(value: dict[str, Any] | None, fallback: VariantConfig) -> VariantConfig:
    if not isinstance(value, dict):
        return fallback
    return VariantConfig(
        name=str(value.get("name") or fallback.name),
        use_multi_step_reasoning=bool(value.get("use_multi_step_reasoning", fallback.use_multi_step_reasoning)),
        use_critic_revision=bool(value.get("use_critic_revision", fallback.use_critic_revision)),
        use_lyric_retriever=bool(value.get("use_lyric_retriever", fallback.use_lyric_retriever)),
    )


def run_chat_query(prompt: str, variant_config: VariantConfig) -> dict[str, Any]:
    songs, retrieval_prompt_config, embedder, lyric_index = load_app_resources()
    context = create_graph_runtime_context(
        songs=songs,
        retrieval_prompt_config=retrieval_prompt_config,
        lyric_index=lyric_index,
        lyric_embedder=embedder,
        variant_config=variant_config,
    )
    logger.info(
        "recommendation_graph_run_start",
        query=prompt,
        variant_name=variant_config.name,
    )
    state = run_recommendation(prompt, context)
    reliability = state.get("reliability") or {}
    logger.info(
        "recommendation_graph_run_finished",
        candidate_count=len(state.get("candidate_ids", ())),
        scored_count=len(state.get("scored_candidates", ())),
        explanation_count=len(state.get("explanations", ())),
        confidence=reliability.get("confidence_label") if isinstance(reliability, dict) else None,
        warning_count=len(state.get("warnings", ())),
    )
    return format_graph_response(state, songs)


def format_graph_response(state: dict, songs) -> dict[str, Any]:
    explanations = tuple(state.get("explanations", ()))
    scored_candidates = tuple(state.get("scored_candidates", ()))
    reliability = state.get("reliability") or {}
    warnings = tuple(state.get("warnings", ()))
    record_lookup = _song_record_lookup(songs)

    confidence = reliability.get("confidence_label") if isinstance(reliability, dict) else None
    confidence_score = reliability.get("confidence_score") if isinstance(reliability, dict) else None
    recommendations: list[dict[str, Any]] = []

    if explanations:
        for explanation in explanations:
            track_id = str(explanation.get("track_id", ""))
            recommendations.append(
                _recommendation_display_record(
                    track_id=track_id,
                    rank=explanation.get("rank", len(recommendations) + 1),
                    explanation=explanation.get("explanation", "No explanation was returned."),
                    record_lookup=record_lookup,
                    scored_candidates=scored_candidates,
                )
            )
    elif scored_candidates:
        for candidate in scored_candidates:
            track_id = str(candidate.get("track_id", ""))
            recommendations.append(
                _recommendation_display_record(
                    track_id=track_id,
                    rank=candidate.get("rank", len(recommendations) + 1),
                    explanation=None,
                    record_lookup=record_lookup,
                    scored_candidates=scored_candidates,
                )
            )

    reliability_warnings = ()
    if isinstance(reliability, dict):
        reliability_warnings = tuple(reliability.get("warnings", ()))
    all_warnings = _dedupe_strings((*warnings, *reliability_warnings))
    return {
        "confidence_label": confidence,
        "confidence_score": confidence_score,
        "recommendations": recommendations[:MAX_UI_RECOMMENDATIONS],
        "warnings": list(all_warnings[:8]),
    }


def _recommendation_display_record(
    *,
    track_id: str,
    rank: Any,
    explanation: Any,
    record_lookup: dict[str, dict],
    scored_candidates: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    scored = next(
        (candidate for candidate in scored_candidates if isinstance(candidate, dict) and str(candidate.get("track_id")) == track_id),
        {},
    )
    record = record_lookup.get(track_id, {})
    return {
        "rank": rank,
        "track_id": track_id,
        "title": _track_title(track_id, record_lookup),
        "track_name": _clean_display_value(record.get("track_name")),
        "artist": _clean_display_value(record.get("track_artist")),
        "confidence": scored.get("final_score"),
        "explanation": _clean_display_value(explanation) or "Final score is based on retrieved metadata, lyric evidence, and audio-feature fit.",
    }


def _assistant_message_html(content: Any) -> str:
    if isinstance(content, dict):
        return _result_card_html(content)
    return f'<div class="vf-message">{html.escape(str(content))}</div>'


def _mark_scroll_target(markup: str) -> str:
    if 'data-scroll-target="true"' in markup:
        return markup
    return markup.replace("<div", '<div data-scroll-target="true"', 1)


def _result_card_html(result: dict[str, Any]) -> str:
    confidence_score = result.get("confidence_score")
    confidence_label = result.get("confidence_label")
    if confidence_score is not None:
        confidence_text = f"{float(confidence_score):.2f}"
    elif confidence_label:
        confidence_text = str(confidence_label)
    else:
        confidence_text = "unknown"

    parts = [f'<div class="vf-result-card"><div class="vf-result-title">Results: Confidence: {html.escape(confidence_text)}</div>']
    recommendations = result.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            parts.append(_track_result_html(item))
    else:
        parts.append("<p>I could not produce grounded recommendations from the dataset for that request.</p>")

    warnings = result.get("warnings") or []
    if warnings:
        warning_items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
        parts.append(f'<div class="vf-warnings"><div class="vf-warnings-title">Warnings</div><ul>{warning_items}</ul></div>')
    parts.append("</div>")
    return "".join(parts)


def _track_result_html(item: dict[str, Any]) -> str:
    rank = html.escape(str(item.get("rank", "?")))
    title = html.escape(str(item.get("title") or item.get("track_id") or "Unknown track"))
    track_id = str(item.get("track_id") or "").strip()
    confidence = item.get("confidence")
    confidence_text = f"{float(confidence):.2f}" if confidence is not None else "not available"
    explanation = html.escape(str(item.get("explanation") or "No explanation was returned."))
    iframe = _spotify_iframe_html(track_id)
    return (
        '<div class="vf-track">'
        f'<div class="vf-track-heading">{rank}. {title}</div>'
        f'<div class="vf-track-confidence">Confidence: {html.escape(confidence_text)}</div>'
        f"{iframe}"
        f'<div class="vf-why"><strong>Why:</strong> {explanation}</div>'
        "</div>"
    )


def _spotify_iframe_html(track_id: str) -> str:
    safe_track_id = html.escape(track_id, quote=True)
    if not safe_track_id:
        return ""
    src = f"https://open.spotify.com/embed/track/{safe_track_id}"
    return (
        '<iframe class="vf-spotify-iframe" '
        f'src="{src}" '
        'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
        'loading="lazy"></iframe>'
    )


def _song_record_lookup(songs) -> dict[str, dict]:
    if "track_id" not in songs.columns:
        return {}
    lookup = {}
    for _, row in songs.iterrows():
        track_id = str(row["track_id"])
        if track_id not in lookup:
            lookup[track_id] = row.to_dict()
    return lookup


def _track_title(track_id: str, record_lookup: dict[str, dict]) -> str:
    record = record_lookup.get(track_id, {})
    track_name = _clean_display_value(record.get("track_name"))
    artist = _clean_display_value(record.get("track_artist"))
    if track_name and artist:
        return f"{track_name} - {artist}"
    if track_name:
        return track_name
    if artist:
        return f"{track_id} - {artist}"
    return track_id


def _clean_display_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            deduped.append(text)
            seen.add(text)
    return tuple(deduped)


def main() -> None:
    st.set_page_config(page_title="VibeFinder AI", layout="wide")
    render_styles()
    logger.info("app_startup")

    try:
        load_app_resources()
    except Exception as exc:
        logger.exception("startup_resource_load_failed", log_file=str(LOG_FILE))
        st.error(f"Could not load startup resources: {exc}")
        st.caption(f"Details were logged to {LOG_FILE}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "pending_variant_config" not in st.session_state:
        st.session_state.pending_variant_config = None

    is_searching = bool(st.session_state.pending_query)

    main_left, main_right = st.columns([1.35, 5.0], gap="small")
    with main_left:
        variant_config = render_variant_controls()
    with main_right:
        render_chat_panel(is_searching=is_searching)

    with st.form("chat_form", clear_on_submit=True):
        input_col, button_col = st.columns([12, 1])
        with input_col:
            prompt = st.text_input(
                "Music request",
                placeholder=CHAT_PLACEHOLDER,
                label_visibility="collapsed",
                disabled=is_searching,
            )
        with button_col:
            submitted = st.form_submit_button("➤", disabled=is_searching)

    if submitted and prompt.strip():
        submitted_prompt = prompt.strip()
        logger.info("user_prompt_received", character_count=len(submitted_prompt))
        st.session_state.messages.append({"role": "user", "content": submitted_prompt})
        st.session_state.pending_query = submitted_prompt
        st.session_state.pending_variant_config = _variant_to_session_dict(variant_config)
        st.rerun()

    if st.session_state.pending_query:
        submitted_prompt = str(st.session_state.pending_query)
        pending_variant = _variant_from_session_dict(st.session_state.pending_variant_config, variant_config)
        try:
            response = run_chat_query(submitted_prompt, pending_variant)
        except Exception as exc:
            logger.exception(
                "recommendation_graph_run_failed",
                variant_name=pending_variant.name,
                log_file=str(LOG_FILE),
            )
            response = f"Could not complete the recommendation run: {exc}\n\nDetails were logged to {LOG_FILE}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.pending_query = None
        st.session_state.pending_variant_config = None
        st.rerun()


if __name__ == "__main__":
    main()
