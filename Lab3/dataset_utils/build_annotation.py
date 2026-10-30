# build_annotation.py
import os
from pathlib import Path
import pandas as pd


def extract_class_from_filename(filename: str) -> str:
    """Извлекает название класса из имени файла.

    Примеры:
        zebra_0001.jpg        -> "zebra"
        bay_horse_0002.jpg    -> "bay_horse"
        bay horse_0001.jpg    -> "bay horse"
    """
    stem = Path(filename).stem
    if "_" in stem:
        base, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            return base.strip()
    return stem.rstrip("0123456789").strip("_- ")


def build_annotation(dataset_dir, annotation_file, project_root=None):
    """Строит CSV-аннотацию для всех файлов в dataset_dir.

    :param dataset_dir: путь к папке датасета
    :param annotation_file: путь к итоговому CSV
    :param project_root: база для relative_path (по умолчанию — родитель annotation_file)
    :return: pandas.DataFrame
    """
    dataset_dir = Path(dataset_dir).resolve()
    annotation_file = Path(annotation_file).resolve()
    project_root = Path(project_root).resolve() if project_root else annotation_file.parent

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {dataset_dir}")

    rows = []
    for file_path in sorted(dataset_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.resolve() == annotation_file:
            continue

        abs_path = file_path.resolve()
        try:
            rel_path = abs_path.relative_to(project_root)
        except ValueError:
            rel_path = os.path.relpath(abs_path, project_root)

        rows.append({
            "absolute_path": str(abs_path),
            "relative_path": str(rel_path),
            "class_label": extract_class_from_filename(file_path.name),
        })

    df = pd.DataFrame(rows, columns=["absolute_path", "relative_path", "class_label"])
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(annotation_file, index=False, sep=";", encoding="utf-8-sig")

    print(f"Аннотация записана: {annotation_file}")
    print(f"Всего файлов: {len(df)}")
    for label, count in df["class_label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")

    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build_annotation(root / "dataset", root / "annotation.csv", project_root=root)