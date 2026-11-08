import os
from pathlib import Path


def rename_files_with_underscores(folder_path: str) -> None:
    """
    Переименовывает все файлы в указанной папке,
    заменяя пробелы на подчёркивания.
    """
    folder = Path(folder_path)

    if not folder.is_dir():
        print(f"Ошибка: '{folder_path}' — это не папка или её не существует.")
        return

    renamed_count = 0

    for file_path in folder.iterdir():
        # Пропускаем директории (работаем только с файлами)
        if not file_path.is_file():
            continue

        # Если в имени нет пробелов — пропускаем
        if " " not in file_path.name:
            continue

        new_name = file_path.name.replace(" ", "_")
        new_path = file_path.with_name(new_name)

        # Проверка на конфликт имён
        if new_path.exists():
            print(f"Пропущено (файл уже существует): {new_name}")
            continue

        try:
            file_path.rename(new_path)
            print(f"Переименован: '{file_path.name}' -> '{new_name}'")
            renamed_count += 1
        except OSError as e:
            print(f"Ошибка при переименовании '{file_path.name}': {e}")

    print(f"\nГотово. Переименовано файлов: {renamed_count}")


if __name__ == "__main__":
    # Укажите путь к папке здесь или передайте аргументом командной строки
    import sys

    target_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    rename_files_with_underscores(target_folder)