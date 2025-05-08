from pathlib import Path
import sys

import pytest

TESTS = Path(__file__).parent.resolve()
ROOT = TESTS.parent.resolve()
SRC = ROOT / "FedBatchDesigner"

# add to `sys.path` so that we can import modules from `FedBatchDesigner`
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def test_data_dir():
    return TESTS / "data"
