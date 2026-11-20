import pytest
import shuffle_dataset as sd


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sd, "SEED", 42)
    monkeypatch.setattr(sd, "COPY_FILES", True)
    monkeypatch.setattr(sd, "NUM_RANGE", (0, 100))
    return tmp_path


def test_shuffle_positive(patched):
    ds = patched / "data"
    ds.mkdir()
    (ds / "zebra_0001.jpg").write_bytes(b"1")
    (ds / "zebra_0002.jpg").write_bytes(b"2")
    (ds / "bay_horse_0001.jpg").write_bytes(b"3")

    df = sd.shuffle_dataset("data", "shuffled", "shuf.csv")
    assert len(df) == 3
    assert (patched / "shuffled.csv" if False else patched / "shuf.csv").exists()
    assert (patched / "shuffled").exists()
    assert len(list((patched / "shuffled").iterdir())) == 3
    # имена — просто числа с расширением
    for p in (patched / "shuffled").iterdir():
        assert p.stem.isdigit()
    # источник не тронут (COPY_FILES=True)
    assert len(list(ds.iterdir())) == 3
    # класс восстановлен по исходному имени
    assert set(df["class_label"]) == {"zebra", "bay_horse"}
    # номера уникальны
    assert df["number"].is_unique


def test_shuffle_missing_dir(patched):
    with pytest.raises(FileNotFoundError):
        sd.shuffle_dataset("nope", "shuffled", "s.csv")


def test_shuffle_empty_dir(patched):
    (patched / "empty").mkdir()
    with pytest.raises(ValueError, match="нет файлов"):
        sd.shuffle_dataset("empty", "shuffled", "s.csv")


def test_shuffle_not_enough_numbers(patched, monkeypatch):
    monkeypatch.setattr(sd, "NUM_RANGE", (0, 1))  # только 2 номера
    ds = patched / "data"
    ds.mkdir()
    for i in range(5):
        (ds / f"zebra_{i:04d}.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="больше"):
        sd.shuffle_dataset("data", "shuffled", "s.csv")


def test_shuffle_move_mode(patched, monkeypatch):
    monkeypatch.setattr(sd, "COPY_FILES", False)
    ds = patched / "data"
    ds.mkdir()
    (ds / "zebra_0001.jpg").write_bytes(b"1")
    (ds / "zebra_0002.jpg").write_bytes(b"2")
    sd.shuffle_dataset("data", "shuffled", "s.csv")
    # источник пуст
    assert list(ds.iterdir()) == []
    assert len(list((patched / "shuffled").iterdir())) == 2