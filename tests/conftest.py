"""Suite-wide guards: no test may read the developer's ``.env``, its release remote or its GitHub token.

The autouse fixture points ``NO_QUIZ_ENV_FILE`` at a file that does not exist, so ``load_project_env`` never reads the
repository's real ``.env`` (a test that needs one sets its own fake file, which overrides this). It also sets
``testkit.HERMETIC_RELEASE_ENV`` (remote URL and token empty, branch ``main``) in the environment of every test.
``load_project_env`` never replaces a variable that is already set, even an empty one, so in-process
``RuntimeSettings.from_env()`` sees no remote and every deploy goes to the workspace's local bare remote.
Spawned processes do not inherit this environment by themselves (the MCP SDK starts stdio servers with a minimal
one): each launch path passes the same variables explicitly (``STDIO_ENV``, ``server_parameters``, the bridge).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from testkit import HERMETIC_RELEASE_ENV

from no_quiz_no_prod.contracts import EnvVar


@pytest.fixture(autouse=True)
def hermetic_release_remote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv(EnvVar.NO_QUIZ_ENV_FILE.value, str(tmp_path / "no-such.env"))  # never created
    for name, value in HERMETIC_RELEASE_ENV.items():
        monkeypatch.setenv(name, value)
    yield
