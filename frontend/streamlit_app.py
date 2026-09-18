import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Darukaa Biodiversity AI")
st.title("Darukaa Biodiversity AI")

if st.button("Check API health"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        st.json(response.json())
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
