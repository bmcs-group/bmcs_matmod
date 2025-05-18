import pytest
import numpy as np

@pytest.fixture
def standard_time_array():
    """Return a standard time array for testing."""
    return np.linspace(0, 10, 100)
