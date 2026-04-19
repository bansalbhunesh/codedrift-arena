"""
Hugging Face Spaces entrypoint (repo root).

HF loads ``app.py`` and the ``demo`` object by convention for Gradio Spaces.
"""

from hf_space.space_app import demo

__all__ = ["demo"]
