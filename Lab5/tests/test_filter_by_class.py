import pandas as pd
import pytest
from filter_by_class import filter_by_class, filter_by_class_and_size


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "class_label": ["zebra", "zebra", "bay_horse", "bay_horse", "zebra"],
            "height": [10, 20, 30, 40, 50],
            "width": [10, 20, 30, 40, 50],
        }
    )


def test_filter_by_class_positive(df):
    r = filter_by_class(df, "zebra")
    assert len(r) == 3
    assert (r["class_label"] == "zebra").all()


def test_filter_by_class_unknown_label_returns_empty(df):
    r = filter_by_class(df, "elephant")
    assert r.empty
    assert isinstance(r, pd.DataFrame)


def test_filter_returns_copy_not_view(df):
    r = filter_by_class(df, "zebra")
    r.loc[r.index[0], "class_label"] = "changed"
    assert (df["class_label"] == "zebra").sum() == 3


def test_filter_by_class_and_size_positive(df):
    r = filter_by_class_and_size(df, "zebra", max_height=20, max_width=20)
    assert len(r) == 2
    assert (r["height"] <= 20).all() and (r["width"] <= 20).all()


def test_filter_by_class_and_size_negative_limits_return_empty(df):
    r = filter_by_class_and_size(df, "zebra", max_height=-1, max_width=-1)
    assert r.empty


def test_filter_by_class_none_label_returns_empty(df):
    # pandas молча вернёт пустой результат — никакого исключения
    assert filter_by_class(df, None).empty


def test_filter_by_class_and_size_missing_columns_raises():
    bad = pd.DataFrame({"class_label": ["zebra"]})
    with pytest.raises(KeyError):
        filter_by_class_and_size(bad, "zebra", max_height=10, max_width=10)