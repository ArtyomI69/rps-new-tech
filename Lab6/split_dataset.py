import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Загружаем аннотацию
df = pd.read_csv(
    "annotation.csv",
    sep=";",
    encoding="utf-8-sig",   # убирает BOM в начале файла
)

# На всякий случай чистим пробелы в названиях колонок
df.columns = df.columns.str.strip()

print("Всего изображений:", len(df))
print("Распределение по классам:\n", df["class_label"].value_counts())

# 2. Первое разбиение: train (например 70%) vs temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class_label"],   # сохраняем пропорции классов
    shuffle=True,
)

# 3. Второе разбиение: temp -> val (15%) и test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,               # половина от 30% = 15%
    random_state=42,
    stratify=temp_df["class_label"],
    shuffle=True,
)

# 4. Проверка результатов
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n=== {name} ({len(part)} шт., {len(part)/len(df)*100:.1f}%) ===")
    print(part["class_label"].value_counts())

# 5. Сохраняем разбиения в отдельные CSV
train_df.to_csv("train.csv", sep=";", index=False, encoding="utf-8-sig")
val_df.to_csv("val.csv",   sep=";", index=False, encoding="utf-8-sig")
test_df.to_csv("test.csv", sep=";", index=False, encoding="utf-8-sig")

print("\nФайлы train.csv / val.csv / test.csv сохранены.")