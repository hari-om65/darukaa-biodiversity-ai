import os
import uuid
from typing import Any

import requests
import streamlit as st

# st.secrets raises rather than returning a default when no secrets.toml
# exists at all (e.g. local dev without one configured), so this can't just
# be st.secrets.get(...).
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

CHAT_TIMEOUT = 60  # the backend calls an LLM for /chat and /chat/structured
FAST_TIMEOUT = 15  # /whatif and /health do no LLM call

_HORIZON_COLORS = {"short": "green", "medium": "orange", "long": "gray"}
_CONFIDENCE_COLORS = {"low": "red", "medium": "orange", "high": "green"}
_CROP_OPTIONS = ["monoculture", "agroforestry", "intercropping", "polyculture", "crop_rotation", "fallow"]
_RAINFALL_OPTIONS = ["low", "medium", "high"]
_WHATIF_VARIABLES = ["soil_organic_carbon", "rainfall", "crop", "region"]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history: list[dict[str, Any]] = []
if "pending_error" not in st.session_state:
    st.session_state.pending_error = None
if "last_whatif" not in st.session_state:
    st.session_state.last_whatif = None


def _new_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.pending_error = None
    st.session_state.last_whatif = None


# --------------------------------------------------------------------------
# Backend calls
# --------------------------------------------------------------------------


def _post(url: str, payload: dict[str, Any], timeout: int = CHAT_TIMEOUT) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _error_message(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail")
        except Exception:
            detail = None
        if detail:
            return str(detail)
        return f"Backend returned {exc.response.status_code}."
    return f"Could not reach the backend at {BACKEND_URL}: {exc}"


def _assistant_entry_from_response(data: dict[str, Any]) -> dict[str, Any]:
    recommendations = (data.get("recommendations") or {}).get("recommendations", [])
    return {
        "role": "assistant",
        "content": data.get("reply", ""),
        "status": data.get("status"),
        "missing_fields": data.get("missing_fields") or [],
        "recommendations": recommendations,
        "explain": data.get("explain"),
    }


def _submit_turn(url: str, payload: dict[str, Any], user_content: str) -> None:
    """Post one conversational turn, append both sides to history, and rerun
    so the top-level history loop redraws with the new messages included."""
    st.session_state.history.append({"role": "user", "content": user_content})
    with st.spinner("Thinking..."):
        try:
            data = _post(url, payload)
        except requests.RequestException as exc:
            st.session_state.pending_error = _error_message(exc)
        else:
            st.session_state.history.append(_assistant_entry_from_response(data))
    st.rerun()


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _has_explain_content(explain: dict[str, Any] | None) -> bool:
    if not explain:
        return False
    return any(explain.get(key) for key in ("thresholds_fired", "graph_paths", "evidence", "recommendation_mapping"))


def _render_recommendation_cards(recommendations: list[dict[str, Any]]) -> None:
    for rec in recommendations:
        with st.container(border=True):
            st.markdown(f"**{rec.get('action', '')}**")
            st.write(rec.get("mechanism", ""))

            cols = st.columns(3)
            with cols[0]:
                st.badge(str(rec.get("metric_impacted", "")), color="blue")
            with cols[1]:
                horizon = rec.get("time_horizon", "")
                st.badge(f"{horizon}-term", color=_HORIZON_COLORS.get(horizon, "gray"))
            with cols[2]:
                confidence = rec.get("confidence", "")
                st.badge(f"confidence: {confidence}", color=_CONFIDENCE_COLORS.get(confidence, "gray"))

            sources = rec.get("sources") or []
            if sources:
                st.caption("Sources")
                for source in sources:
                    st.markdown(f"- {source}")


def _truncate(text: str, limit: int = 220) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _render_explain(explain: dict[str, Any] | None) -> None:
    if not _has_explain_content(explain):
        return

    with st.expander("Why this answer"):
        thresholds = explain.get("thresholds_fired") or []
        if thresholds:
            st.markdown("**Thresholds triggered**")
            for t in thresholds:
                st.markdown(f"- `{t['variable']} {t['operator']} {t['value']}` → **{t['flag']}**")

        graph_paths = explain.get("graph_paths") or []
        if graph_paths:
            st.markdown("**Reasoning chains walked**")
            for path in graph_paths:
                st.markdown("- " + " → ".join(path))

        evidence = explain.get("evidence") or []
        if evidence:
            st.markdown("**Evidence retrieved**")
            for chain_evidence in evidence:
                path_label = " → ".join(chain_evidence.get("path", []))
                st.markdown(f"*Chain: {path_label}*")
                for chunk in chain_evidence.get("chunks", []):
                    score = chunk.get("score")
                    score_label = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                    st.markdown(
                        f"> **{chunk.get('source')}** ({chunk.get('year')}, relevance {score_label}) — "
                        f"{_truncate(chunk.get('text', ''))}"
                    )

        mapping = explain.get("recommendation_mapping") or []
        if mapping:
            st.markdown("**How recommendations use this evidence**")
            for m in mapping:
                chains_label = "; ".join(" → ".join(c) for c in m.get("supporting_chains", []))
                sources_label = ", ".join(m.get("sources", []))
                st.markdown(f"- *{m.get('action')}* — via {chains_label or 'no chain'}; sources: {sources_label}")


def _render_chain_list(chains: list[dict[str, Any]], empty_label: str = "None") -> None:
    if not chains:
        st.caption(empty_label)
        return
    for chain in chains:
        path = " → ".join(chain.get("path", []))
        relation = chain.get("relation", "")
        strength = chain.get("strength", "")
        horizon = chain.get("projected_horizon", "")
        st.markdown(f"- {path}  \n  _{relation}, {strength} strength, {horizon}-term_")


def _render_whatif_result(data: dict[str, Any]) -> None:
    st.markdown(f"**{data.get('variable')}**: `{data.get('old_value')}` → `{data.get('new_value')}`")

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Before**")
        _render_chain_list(data.get("before_chains", []))
    with col_after:
        st.markdown("**After**")
        _render_chain_list(data.get("after_chains", []))

    removed = data.get("removed_chains", [])
    added = data.get("added_chains", [])
    if removed:
        st.markdown("**🔴 Chains removed**")
        _render_chain_list(removed)
    if added:
        st.markdown("**🟢 Chains added**")
        _render_chain_list(added)
    if not removed and not added:
        st.caption("No change in triggered reasoning chains.")

    _render_explain(data.get("explain"))


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Darukaa Biodiversity AI", layout="wide")

st.title("Darukaa Biodiversity AI")
st.caption(f"Backend: {BACKEND_URL}")

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("New conversation"):
        _new_conversation()
        st.rerun()

    st.divider()
    st.subheader("Backend status")
    if st.button("Check API health"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=FAST_TIMEOUT)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(_error_message(exc))

if st.session_state.pending_error:
    st.error(st.session_state.pending_error)
    st.session_state.pending_error = None

with st.expander("Structured input mode"):
    st.caption("Already know the site data? Skip the conversation and submit it directly.")

    soc_percent = st.number_input(
        "Soil organic carbon (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1,
        help='Sent to the backend as a fraction, e.g. 30% -> 0.3.',
    )
    rainfall = st.selectbox("Rainfall", _RAINFALL_OPTIONS, key="structured_rainfall")
    crop = st.selectbox("Crop / land use", _CROP_OPTIONS, key="structured_crop")
    region = st.text_input("Region", placeholder="e.g. semi-arid", key="structured_region")

    if st.button("Submit structured input"):
        if not region.strip():
            st.warning("Please enter a region.")
        else:
            inputs = {
                "soil_organic_carbon": round(soc_percent / 100, 4),
                "rainfall": rainfall,
                "crop": crop,
                "region": region.strip(),
            }
            _submit_turn(
                f"{BACKEND_URL}/chat/structured",
                {"session_id": st.session_state.session_id, "inputs": inputs},
                user_content=(
                    f"[Structured input] soil_organic_carbon={soc_percent}%, "
                    f"rainfall={rainfall}, crop={crop}, region={region.strip()}"
                ),
            )

with st.expander("What-if simulator"):
    st.caption(
        "Propose a change to one variable and see how the reasoning chains shift, "
        "without changing your actual session data."
    )

    whatif_variable = st.selectbox("Variable to change", _WHATIF_VARIABLES, key="whatif_variable")

    if whatif_variable == "soil_organic_carbon":
        whatif_percent = st.number_input(
            "New soil organic carbon (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1, key="whatif_soc"
        )
        whatif_value: Any = round(whatif_percent / 100, 4)
    elif whatif_variable == "rainfall":
        whatif_value = st.selectbox("New rainfall", _RAINFALL_OPTIONS, key="whatif_rainfall")
    elif whatif_variable == "crop":
        whatif_value = st.selectbox("New crop / land use", _CROP_OPTIONS, key="whatif_crop")
    else:
        whatif_value = st.text_input("New region", key="whatif_region")

    if st.button("Simulate"):
        if whatif_variable == "region" and not str(whatif_value).strip():
            st.warning("Please enter a region.")
        else:
            with st.spinner("Simulating..."):
                try:
                    st.session_state.last_whatif = _post(
                        f"{BACKEND_URL}/whatif",
                        {
                            "session_id": st.session_state.session_id,
                            "variable": whatif_variable,
                            "value": whatif_value,
                        },
                        timeout=FAST_TIMEOUT,
                    )
                except requests.RequestException as exc:
                    st.error(_error_message(exc))

    if st.session_state.last_whatif:
        st.divider()
        _render_whatif_result(st.session_state.last_whatif)

st.divider()

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("status") == "needs_clarification" and entry.get("missing_fields"):
            st.caption("Still need: " + ", ".join(entry["missing_fields"]))
        if entry.get("recommendations"):
            _render_recommendation_cards(entry["recommendations"])
        if entry.get("explain") is not None:
            _render_explain(entry["explain"])

prompt = st.chat_input("Tell me about your site, or ask a question...")
if prompt:
    _submit_turn(
        f"{BACKEND_URL}/chat",
        {"session_id": st.session_state.session_id, "message": prompt},
        user_content=prompt,
    )
