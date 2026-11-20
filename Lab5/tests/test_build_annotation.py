import pytest
import build_annotation as ba


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_extract_class_from_filename_variants():
    assert ba.extract_class_from_filename("zebra_0001.jpg") == "zebra"
    assert ba.extract_class_from_filename("bay_horse_0002.jpg") == "bay_horse"
    assert ba.extract_class_from_filename("bay horse_0001.jpg") == "bay horse"
    assert ba.extract_class_from_filename("camel.png") == "camel"
    assert ba.extract_class_from_filename("no_suffix_xyz.jpg") == "no_suffix_xyz"


def test_build_annotation_positive(patched):
    ds = patched / "data"
    ds.mkdir()
    (ds / "zebra_0001.jpg").write_bytes(b"x")
    (ds / "zebra_0002.jpg").write_bytes(b"x")
    (ds / "bay_horse_0001.jpg").write_bytes(b"x")

    df = ba.build_annotation("data", "ann.csv")
    assert len(df) == 3
    assert set(df["class_label"]) == {"zebra", "bay_horse"}
    assert (patched / "ann.csv").exists()
    # CSV читается без BOM-порчи имени колонки
    import pandas as pd

    check = pd.read_csv(patched / "ann.csv", sep=";", encoding="utf-8-sig")
    assert list(check.columns) == ["absolute_path", "relative_path", "class_label"]


def test_build_annotation_missing_dir(patched):
    with pytest.raises(FileNotFoundError):
        ba.build_annotation("nope", "ann.csv")


def test_build_annotation_ignores_hidden_files(patched):
    ds = patched / "data"
    ds.mkdir()
    (ds / "zebra_0001.jpg").write_bytes(b"x")
    (ds / ".hidden.jpg").write_bytes(b"x")
    df = ba.build_annotation("data", "ann.csv")
    assert len(df) == 1


def test_build_annotation_empty_dir(patched):
    ds = patched / "empty"
    ds.mkdir()
    df = ba.build_annotation("empty", "ann.csv")
    assert len(df) == 0
    assert list(df.columns) == ["absolute_path", "relative_path", "class_label"]


def test_build_annotation_excludes_self_csv(patched):
    ds = patched / "data"
    ds.mkdir()
    (ds / "zebra_0001.jpg").write_bytes(b"x")
    # аннотация внутри датасета — не должна попасть в саму себя
    df = ba.build_annotation("data", "data/ann.csv")
    assert len(df) == 1