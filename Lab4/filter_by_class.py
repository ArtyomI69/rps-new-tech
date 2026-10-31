import pandas as pd

def filter_by_class(df: pd.DataFrame, class_label: str) -> pd.DataFrame:
    """
    Фильтрует DataFrame по метке класса.

    Параметры
    ---------
    df : pd.DataFrame
        Исходный DataFrame (должен содержать колонку 'class_label').
    class_label : str
        Метка класса, по которой нужно отфильтровать строки
        (например, 'bay horse' или 'zebra').

    Возвращает
    ----------
    pd.DataFrame
        Новый DataFrame, содержащий только строки с заданной меткой класса.
    """
    return df[df['class_label'] == class_label].copy()


def filter_by_class_and_size(
    df: pd.DataFrame,
    class_label: str,
    max_height: float,
    max_width: float
) -> pd.DataFrame:
    """
    Фильтрует DataFrame по метке класса и максимальным размерам изображения.

    Условие: height <= max_height AND width <= max_width
             AND class_label == class_label.

    Параметры
    ---------
    df : pd.DataFrame
        Исходный DataFrame (должен содержать колонки
        'class_label', 'height', 'width').
    class_label : str
        Метка класса для фильтрации.
    max_height : float
        Максимально допустимая высота изображения.
    max_width : float
        Максимально допустимая ширина изображения.

    Возвращает
    ----------
    pd.DataFrame
        Новый DataFrame с отфильтрованными строками.
    """
    mask = (
        (df['class_label'] == class_label)
        & (df['height'] <= max_height)
        & (df['width'] <= max_width)
    )
    return df[mask].copy()