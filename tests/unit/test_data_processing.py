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
        1000: 2,
        1001: 3,
        1002: 4,
        1003: 5,
        1004: 6,
        1005: 7,
        1006: 8,
        1007: 9,
        1008: 10,
        1009: 11,
    }
    assert inverse_mapping == {
        2: 1000,
        3: 1001,
        4: 1002,
        5: 1003,
        6: 1004,
        7: 1005,
        8: 1006,
        9: 1007,
        10: 1008,
        11: 1009,
    }
