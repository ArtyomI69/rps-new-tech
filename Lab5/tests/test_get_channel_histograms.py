import numpy as np
import pandas as pd
import pytest
from get_channel_histograms import get_channel_histograms


@pytest.fixture
def df_single_rgb(sample_image_path):
    return pd.DataFrame(
        {"class_label": ["zebra"], "absolute_path": [str(sample_image_path)]}
    )


@pytest.fixture
def df_single_gray(sample_gray_image_path):
    return pd.DataFrame(
        {"class_label": ["zebra"], "absolute_path": [str(sample_gray_image_path)]}
    )


def test_histograms_positive(df_single_rgb):
    hb, hg, hr = get_channel_histograms(df_single_rgb, "zebra", bins=256, seed=0)
    assert hb.shape == hg.shape == hr.shape == (256,)
    # нормировка
    assert np.isclose(hb.sum(), 1.0)
    assert np.isclose(hg.sum(), 1.0)
    assert np.isclose(hr.sum(), 1.0)
    # пики каналов: B=255, G=100, R=50
    assert hb.argmax() == 255
    assert hg.argmax() == 100
    assert hr.argmax() == 50


def test_histograms_grayscale_image(df_single_gray):
    # не должно упасть: grayscale конвертируется в BGR
    hb, hg, hr = get_channel_histograms(df_single_gray, "zebra", seed=0)
    assert np.isclose(hb.sum(), 1.0)


def test_histograms_class_not_found(sample_image_path):
    df = pd.DataFrame(
        {"class_label": ["horse"], "absolute_path": [str(sample_image_path)]}
    )
    with pytest.raises(ValueError, match="нет изображений класса"):
        get_channel_histograms(df, "zebra")


def test_histograms_empty_df():
    df = pd.DataFrame(columns=["class_label", "absolute_path"])
    with pytest.raises(ValueError):
        get_channel_histograms(df, "zebra")


def test_histograms_unreadable_file(tmp_path):
    df = pd.DataFrame(
        {"class_label": ["zebra"], "absolute_path": [str(tmp_path / "missing.jpg")]}
    )
    with pytest.raises(IOError, match="Не удалось прочитать"):
        get_channel_histograms(df, "zebra")


def test_histograms_seed_reproducible(sample_image_path):
    df = pd.DataFrame(
        {"class_label": ["zebra"] * 5, "absolute_path": [str(sample_image_path)] * 5}
    )
    a = get_channel_histograms(df, "zebra", seed=42)[0]
    b = get_channel_histograms(df, "zebra", seed=42)[0]
    assert np.array_equal(a, b)