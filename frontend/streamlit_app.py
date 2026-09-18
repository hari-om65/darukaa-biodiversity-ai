import os

import requests
import streamlit as st


def _get_api_url() -> str:
    # st.secrets raises rather than returning a default when no secrets.toml
    # exists at all (e.g. local dev without one configured), so this can't
    # just be st.secrets.get(...).
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000")


API_URL = _get_api_url()

st.set_page_config(page_title="Darukaa Biodiversity AI")
st.title("Darukaa Biodiversity AI")
st.caption(f"Backend: {API_URL}")

if st.button("Check API health"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
