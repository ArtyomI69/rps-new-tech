import os
import random
import shutil
from pathlib import Path
import pandas as pd

# Корень проекта (папка, где лежит скрипт или сам проект)
PROJECT_ROOT = Path(__file__).resolve().parent

# Исходная папка с датасетом
DATASET_DIR = PROJECT_ROOT / "dataset"

# Куда складывать переименованные файлы
SHUFFLED_DIR = PROJECT_ROOT / "dataset_shuffled"

# Файл-аннотация
ANNOTATION_FILE = PROJECT_ROOT / "annotation_shuffled.csv"

# Диапазон случайных номеров
NUM_RANGE = (0, 2000)

# Seed для воспроизводимости (None — без фиксации)
SEED = 42

# True — копировать, False — перемещать
COPY_FILES = True


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


def shuffle_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {DATASET_DIR}")

    # Пересоздаём целевую папку
    if SHUFFLED_DIR.exists():
        shutil.rmtree(SHUFFLED_DIR)
    SHUFFLED_DIR.mkdir(parents=True, exist_ok=True)

    # Собираем все файлы рекурсивно
    files = []
    for file_path in sorted(DATASET_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.resolve() == ANNOTATION_FILE.resolve():
            continue
        files.append(file_path)

    if not files:
        raise ValueError(f"В папке {DATASET_DIR} нет файлов")

    lo, hi = NUM_RANGE
    capacity = hi - lo + 1
    if len(files) > capacity:
        raise ValueError(
            f"Файлов ({len(files)}) больше, чем доступных номеров ({capacity}). "
            f"Увеличьте NUM_RANGE."
        )

    # Уникальные случайные номера без повторов
    rng = random.Random(SEED)
    numbers = rng.sample(range(lo, hi + 1), len(files))

    rows = []

    for src_path, num in zip(files, numbers):
        new_name = f"{num}{src_path.suffix.lower()}"
        dst_path = SHUFFLED_DIR / new_name

        if COPY_FILES:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(str(src_path), str(dst_path))

        abs_path = dst_path.resolve()
        try:
            rel_path = abs_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = os.path.relpath(abs_path, PROJECT_ROOT)

        class_label = extract_class_from_filename(src_path.name)

        rows.append(
            {
                "absolute_path": str(abs_path),
                "relative_path": str(rel_path),
                "class_label": class_label,
                "original_filename": src_path.name,
                "number": num,
            }
        )

    # Собираем DataFrame
    df = pd.DataFrame(
        rows,
        columns=[
            "absolute_path",
            "relative_path",
            "class_label",
            "original_filename",
            "number",
        ],
    ).sort_values("number").reset_index(drop=True)

    # Записываем CSV (sep=';' + utf-8-sig — корректно открывается в Excel)
    df.to_csv(ANNOTATION_FILE, index=False, sep=";", encoding="utf-8-sig")

    print(f"Датасет переименован: {SHUFFLED_DIR}")
    print(f"Аннотация записана: {ANNOTATION_FILE}")
    print(f"Всего файлов: {len(df)}")

    # Статистика по классам средствами pandas
    print("Распределение по классам:")
    counts = df["class_label"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label}: {count}")

    return df


if __name__ == "__main__":
    shuffle_dataset()