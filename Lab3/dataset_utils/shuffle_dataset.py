# shuffle_dataset.py
import os
import random
import shutil
from pathlib import Path
import pandas as pd

from .build_annotation import extract_class_from_filename


def shuffle_dataset(
    dataset_dir,
    shuffled_dir,
    annotation_file,
    num_range=(0, 2000),
    seed=42,
    copy_files=True,
    project_root=None,
):
    """Создаёт копию/перемещение датасета с числовыми именами и CSV-аннотацией.

    :return: pandas.DataFrame
    """
    dataset_dir = Path(dataset_dir).resolve()
    shuffled_dir = Path(shuffled_dir).resolve()
    annotation_file = Path(annotation_file).resolve()
    project_root = Path(project_root).resolve() if project_root else annotation_file.parent

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {dataset_dir}")

    if shuffled_dir.exists():
        shutil.rmtree(shuffled_dir)
    shuffled_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for file_path in sorted(dataset_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.resolve() == annotation_file:
            continue
        files.append(file_path)

    if not files:
        raise ValueError(f"В папке {dataset_dir} нет файлов")

    lo, hi = num_range
    capacity = hi - lo + 1
    if len(files) > capacity:
        raise ValueError(
            f"Файлов ({len(files)}) больше, чем доступных номеров ({capacity}). "
            f"Увеличьте диапазон."
        )

    rng = random.Random(seed)
    numbers = rng.sample(range(lo, hi + 1), len(files))

    rows = []
    for src_path, num in zip(files, numbers):
        new_name = f"{num}{src_path.suffix.lower()}"
        dst_path = shuffled_dir / new_name

        if copy_files:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(str(src_path), str(dst_path))

        abs_path = dst_path.resolve()
        try:
            rel_path = abs_path.relative_to(project_root)
        except ValueError:
            rel_path = os.path.relpath(abs_path, project_root)

        rows.append({
            "absolute_path": str(abs_path),
            "relative_path": str(rel_path),
            "class_label": extract_class_from_filename(src_path.name),
            "original_filename": src_path.name,
            "number": num,
        })

    df = pd.DataFrame(
        rows,
        columns=["absolute_path", "relative_path", "class_label",
                 "original_filename", "number"],
    ).sort_values("number").reset_index(drop=True)

    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(annotation_file, index=False, sep=";", encoding="utf-8-sig")

    print(f"Датасет переименован: {shuffled_dir}")
    print(f"Аннотация записана: {annotation_file}")
    print(f"Всего файлов: {len(df)}")
    for label, count in df["class_label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")

    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    shuffle_dataset(
        dataset_dir=root / "dataset",
        shuffled_dir=root / "dataset_shuffled",
        annotation_file=root / "annotation_shuffled.csv",
        project_root=root,
    )