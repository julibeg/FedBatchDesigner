import numpy as np
import util


def test_nice_value_range():
    """
    Test the `nice_value_range` function.
    """
    # test with some example values
    assert np.allclose(
        util.nice_value_range(0.12345, 2.8, 0.5), [0.5, 1.0, 1.5, 2.0, 2.5]
    )
