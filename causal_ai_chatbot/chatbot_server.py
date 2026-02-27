"""
Compatibility entrypoint for the chatbot server.

This keeps existing commands (`python chatbot_server.py`) and test imports
working after moving runtime code into `app/`.
"""

from app.chatbot_server import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("app.chatbot_server", run_name="__main__")

