import os
import csv
from pathlib import Path
import pandas as pd

# Корень проекта (папка, где лежит скрипт или сам проект)
PROJECT_ROOT = Path(__file__).resolve().parent

# Папка с датасетом
DATASET_DIR = PROJECT_ROOT / "dataset"

# Файл-аннотация
ANNOTATION_FILE = PROJECT_ROOT / "annotation.csv"


def extract_class_from_filename(filename: str) -> str:
    """
    Извлекает название класса из имени файла.

    Примеры:
        zebra_0001.jpg        -> "zebra"
        bay_horse_0002.jpg    -> "bay_horse"
        bay horse_0001.jpg    -> "bay horse"
    """
    stem = Path(filename).stem  # имя без расширения

    # Отделяем числовой суффикс: ищем последний "_" перед цифрами
    if "_" in stem:
        base, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            return base.strip()

    # Fallback: убираем только цифры в конце
    return stem.rstrip("0123456789").strip("_- ")


def build_annotation():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {DATASET_DIR}")

    rows = []

    # Рекурсивно обходим все файлы в dataset
    for file_path in sorted(DATASET_DIR.rglob("*")):
        if not file_path.is_file():
            continue

        # Пропускаем скрытые файлы и сам csv-файл, если он вдруг попал внутрь
        if file_path.name.startswith("."):
            continue
        if file_path.resolve() == ANNOTATION_FILE.resolve():
            continue

        abs_path = file_path.resolve()

        # Относительный путь относительно корня проекта
        try:
            rel_path = abs_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = os.path.relpath(abs_path, PROJECT_ROOT)

        class_label = extract_class_from_filename(file_path.name)
        rows.append(
            {
                "absolute_path": str(abs_path),
                "relative_path": str(rel_path),
                "class_label": class_label,
            }
        )

    # Собираем DataFrame
    df = pd.DataFrame(rows, columns=["absolute_path", "relative_path", "class_label"])

    # Записываем CSV (sep=';' + utf-8-sig — корректно открывается в Excel)
    df.to_csv(ANNOTATION_FILE, index=False, sep=";", encoding="utf-8-sig")

    print(f"Аннотация записана: {ANNOTATION_FILE}")
    print(f"Всего файлов: {len(df)}")

    # Статистика по классам средствами pandas
    print("Распределение по классам:")
    counts = df["class_label"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label}: {count}")

    return df
