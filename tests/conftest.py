"""Shared test fixtures. All tests use tiny in-memory/tmp-dir fixtures —
no dataset downloads, no index builds beyond a few hundred vectors, no
scientific experiments."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
