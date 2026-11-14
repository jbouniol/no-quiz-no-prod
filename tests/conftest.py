"""Suite-wide guards: no test may read the developer's release remote or GitHub token from ``.env``.

The autouse fixture sets ``testkit.HERMETIC_RELEASE_ENV`` (remote URL and token empty, branch ``main``) in the
environment of every test. ``load_project_env`` never replaces a variable that is already set, even an empty one, so
in-process ``RuntimeSettings.from_env()`` sees no remote and every deploy goes to the workspace's local bare remote.
Spawned processes do not inherit this environment by themselves (the MCP SDK starts stdio servers with a minimal
one): each launch path passes the same variables explicitly (``STDIO_ENV``, ``server_parameters``, the bridge).
"""

from collections.abc import Iterator

import pytest
from testkit import HERMETIC_RELEASE_ENV


@pytest.fixture(autouse=True)
def hermetic_release_remote(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name, value in HERMETIC_RELEASE_ENV.items():
        monkeypatch.setenv(name, value)
    yield
