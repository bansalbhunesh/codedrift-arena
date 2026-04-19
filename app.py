"""
Hugging Face Spaces entrypoint (repo root).

HF loads ``app.py`` and the ``demo`` object by convention for Gradio Spaces.
"""

from codedrift.logutil import configure_logging

configure_logging()

from hf_space.space_app import demo  # noqa: E402

__all__ = ["demo"]
