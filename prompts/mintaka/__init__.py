"""Mintaka-specific prompt templates for all generation regimes."""

from . import prompts as _prompts
from .prompts import *  # noqa: F401,F403

__all__ = getattr(_prompts, "__all__", [])

del _prompts
