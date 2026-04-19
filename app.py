"""
Hugging Face Spaces entrypoint (repo root).

HF loads ``app.py`` and the ``demo`` object by convention for Gradio Spaces.
"""

from codedrift.logutil import configure_logging

configure_logging()

from hf_space.space_app import demo  # noqa: E402

__all__ = ["demo"]

# Hugging Face Spaces runs `python app.py`; without launch() the process exits 0
# immediately after imports and the Space shows a runtime error.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
