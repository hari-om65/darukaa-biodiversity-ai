"""Shared FastAPI dependencies."""

import os

from groq import Groq


def get_groq_client() -> Groq:
    # Groq() raises eagerly at construction if GROQ_API_KEY is unset, unlike
    # the Anthropic SDK, which deferred that check to the first real request.
    # This dependency resolves on every /chat request regardless of whether
    # the handler ends up needing the client (e.g. the needs-clarification
    # and 422-incomplete-input paths never call the API), so a missing key
    # must not fail construction here - it still fails loudly, just on the
    # first actual API call instead.
    return Groq(api_key=os.environ.get("GROQ_API_KEY") or "unset")
