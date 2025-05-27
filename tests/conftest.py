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


@pytest.fixture(scope="session")
def valine_defaults_parsed():
    import params

    parsed = {"common": {}, "s1": {}, "s2": {}}

    for k, v in params.defaults["valine_two_stage"]["values"].items():
        if k.endswith("anaerobic"):
            # this is the anaerobic checkbox --> skip
            continue
        if k.startswith("s1_"):
            parsed["s1"][k[3:]] = round(v, 3)
        elif k.startswith("s2_"):
            parsed["s2"][k[3:]] = round(v, 3)
        else:
            parsed["common"][k] = round(v, 3)

    # apply values of `s1` to `s2` if they are not set in `s2`
    parsed["s2"] = parsed["s1"] | parsed["s2"]

    return parsed
