import shutil
import os


def copy_files(src_dir, dst_dir):
    """
    Использование
    copy_files("/путь/к/источнику", "/путь/к/назначению")
    """
    # Создаём целевую директорию, если её нет
    os.makedirs(dst_dir, exist_ok=True)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 сохраняет метаданные
            print(f"Скопирован: {src_path} -> {dst_path}")
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"Скопирована папка: {src_path} -> {dst_path}")
