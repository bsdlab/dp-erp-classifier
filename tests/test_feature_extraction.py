import pytest

from erp_classifier.context import get_context


@pytest.fixture
def ctx():
    return get_context()


def test_feature_extraction(ctx):
    pass
