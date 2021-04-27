import pandas as pd
import pytest

from recommender.data_processing import map_column


@pytest.fixture
def data():
    df = pd.DataFrame({"id": range(1000, 1010)})
    return df


def test_map_column(data):
    df, mapping, inverse_mapping = map_column(data, col_name="id")

    assert mapping == {
        1000: 1,
        1001: 2,
        1002: 3,
        1003: 4,
        1004: 5,
        1005: 6,
        1006: 7,
        1007: 8,
        1008: 9,
        1009: 10,
    }
    assert inverse_mapping == {
        1: 1000,
        2: 1001,
        3: 1002,
        4: 1003,
        5: 1004,
        6: 1005,
        7: 1006,
        8: 1007,
        9: 1008,
        10: 1009,
    }
