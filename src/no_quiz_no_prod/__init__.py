"""No-Quiz-No-Prod: a pedagogical coding agent that gates production behind tests and comprehension quizzes.

Shared contracts live in ``no_quiz_no_prod.contracts`` and the unified LLM client in ``no_quiz_no_prod.llm``.
Importing the package itself stays side-effect free and cheap.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("no-quiz-no-prod")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
