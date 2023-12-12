import pytest

from noice.main import main, _make_experiment


@pytest.mark.skip(reason="no way we can run this in BitBucket")
def test_dummy_test_to_get_coverage():
    """Test main to get a coverage estimate."""
    main()


def test_make_experiment_smoky():
    """Test main _make_experiment can run."""
    _ = _make_experiment()
