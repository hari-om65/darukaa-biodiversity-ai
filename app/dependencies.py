"""Shared FastAPI dependencies."""

import anthropic


def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()
