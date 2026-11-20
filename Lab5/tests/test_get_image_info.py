import pytest
from get_image_info import get_image_info


def test_rgb_image(sample_image_path):
    h, w, d = get_image_info(str(sample_image_path))
    assert (h, w, d) == (10, 20, 3)


def test_grayscale_image(sample_gray_image_path):
    h, w, d = get_image_info(str(sample_gray_image_path))
    assert (h, w, d) == (15, 25, 1)


def test_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Не удалось открыть"):
        get_image_info(str(tmp_path / "nope.jpg"))


def test_none_path():
    with pytest.raises((ValueError, TypeError)):
        get_image_info(None)


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(ValueError):
        get_image_info(str(tmp_path))  # папка