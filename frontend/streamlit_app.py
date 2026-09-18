import os

import requests
import streamlit as st

# st.secrets raises rather than returning a default when no secrets.toml
# exists at all (e.g. local dev without one configured), so this can't just
# be st.secrets.get(...).
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Darukaa Biodiversity AI")
st.title("Darukaa Biodiversity AI")
st.caption(f"Backend: {BACKEND_URL}")

if st.button("Check API health"):
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
