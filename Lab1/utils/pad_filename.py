def pad_filename(number, total_digits=4, prefix="", extension=""):
    """
    Формирует имя файла с ведущими нулями.
    
    :param number: число, которое нужно дополнить нулями
    :param total_digits: общее количество цифр в номере (по умолчанию 4)
    :param prefix: префикс имени файла (например, "file_")
    :param extension: расширение файла (например, ".txt")
    :return: строка с именем файла
    """
    padded = str(number).zfill(total_digits)
    return f"{prefix}{padded}{extension}"