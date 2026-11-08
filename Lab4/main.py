import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from get_image_info import get_image_info
from get_channel_histograms import get_channel_histograms
from filter_by_class import filter_by_class, filter_by_class_and_size


# ============================================================
# ЗАДАНИЕ 1. Сформировать DataFrame с 2 колонками:
#            «название класса» + «абсолютный путь к файлу».
#            Источник — файл аннотации annotation.csv.
# ============================================================
df = pd.read_csv('annotation.csv', sep=';', encoding='utf-8-sig')
print("[Задание 1] Исходные колонки файла аннотации:")
print(list(df.columns))

# ============================================================
# ЗАДАНИЕ 2. Привести имена колонок к виду:
#            нижний регистр, слова через "_", английский язык.
#            (Если уже так — пункт пропускается, но оставляем
#             нормализацию для надёжности.)
# ============================================================
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
print("\n[Задание 2] Колонки после нормализации имён:")
print(list(df.columns))

# Завершение ЗАДАНИЯ 1: оставляем только две нужные колонки.
df = df[['class_label', 'absolute_path']]
print("\n[Задание 1] Итоговый DataFrame (первые 5 строк):")
print(df.head(5))


# ============================================================
# ЗАДАНИЕ 3. Добавить числовую метку класса:
#            0 — первый класс, 1 — второй.
#            Реализуем через pd.Categorical.
# ============================================================
df['class_id'] = pd.Categorical(
    df['class_label'],
    categories=['bay_horse', 'zebra']
).codes

print("\n" + "=" * 60)
print("[Задание 3] Числовые метки классов:")
print("=" * 60)
print(df[['class_label', 'class_id']].drop_duplicates().to_string(index=False))


# ============================================================
# ЗАДАНИЕ 4. Добавить три столбца:
#            height — высота, width — ширина, depth — глубина (каналы).
#            Значения получаем через OpenCV (get_image_info).
# ============================================================
info = df['absolute_path'].apply(get_image_info)
df[['height', 'width', 'depth']] = pd.DataFrame(info.tolist(), index=df.index)

print("\n" + "=" * 60)
print("[Задание 4] Добавлены столбцы height, width, depth. Пример:")
print("=" * 60)
print(df[['class_label', 'height', 'width', 'depth']].head().to_string(index=False))


# ============================================================
# ЗАДАНИЕ 5. Статистика по размерам изображений (describe)
#            + сводная таблица по меткам (сумма, count)
#            + гистограмма и pie
#            + вывод о сбалансированности по числу элементов в классах.
# ============================================================
print("\n" + "=" * 60)
print("[Задание 5] describe() по размерам изображений:")
print("=" * 60)
size_stats = df[['height', 'width', 'depth']].describe()
print(size_stats)

# Дополнительные характеристики (медиана, мода, размах, IQR)
extra = pd.DataFrame({
    'median': df[['height', 'width', 'depth']].median(),
    'mode':   df[['height', 'width', 'depth']].mode().iloc[0],
    'range':  df[['height', 'width', 'depth']].max() - df[['height', 'width', 'depth']].min(),
    'IQR':    df[['height', 'width', 'depth']].quantile(0.75)
            - df[['height', 'width', 'depth']].quantile(0.25),
})
print("\n[Задание 5] Дополнительные характеристики размеров:")
print(extra)

# Сводная таблица по меткам класса (count / sum / mean)
print("\n[Задание 5] Сводная таблица по меткам класса:")
label_summary = (
    df.groupby('class_label', observed=True)
      .agg(
          count=('class_label', 'size'),
          sum=('class_id', 'sum'),
          height_mean=('height', 'mean'),
          width_mean=('width', 'mean'),
      )
      .reset_index()
)
label_summary['share_%'] = (
    label_summary['count'] / label_summary['count'].sum() * 100
).round(2)

pivot = df.pivot_table(
    index='class_label',
    values=['class_id', 'height', 'width', 'depth'],
    aggfunc=['count', 'sum', 'mean'],
    observed=True,
)
print("\nPivot (count / sum / mean):")
print(pivot)
print("\nСводка по классам:")
print(label_summary.to_string(index=False))

# Визуализация: столбчатая диаграмма + круговая (pie)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_summary.plot(
    kind='bar', x='class_label', y='count',
    ax=axes[0], color=['#4C72B0', '#DD8452'], legend=False,
)
axes[0].set_title('[Задание 5] Количество изображений по классам')
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(label_summary['count']):
    axes[0].text(i, v, str(v), ha='center', va='bottom')

axes[1].pie(
    label_summary['count'],
    labels=label_summary['class_label'],
    autopct='%1.1f%%',
    colors=['#4C72B0', '#DD8452'],
    startangle=90,
    explode=[0.03] * len(label_summary),
)
axes[1].set_title('[Задание 5] Доли классов')

plt.tight_layout()
plt.savefig('class_balance.png', dpi=120)
plt.show()

# Вывод о сбалансированности по числу элементов
counts = label_summary.set_index('class_label')['count']
ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

print("\n[Задание 5] ВЫВОД О СБАЛАНСИРОВАННОСТИ:")
print(f"Количество в классах:\n{counts.to_string()}")
print(f"Соотношение max/min = {ratio:.2f}")
if ratio <= 1.2:
    verdict = "СБАЛАНСИРОВАН (различие ≤ 20%)"
elif ratio <= 1.5:
    verdict = "УМЕРЕННО НЕСБАЛАНСИРОВАН (20–50%)"
else:
    verdict = "НЕСБАЛАНСИРОВАН (> 50%)"
print(f"Итог: {verdict}")


# ============================================================
# ЗАДАНИЕ 6. Функция filter_by_class(df, class_label)
#            — фильтрация DataFrame по метке класса.
#            (реализована в filter_by_class.py, здесь демонстрация)
# ============================================================
print("\n" + "=" * 60)
print("[Задание 6] Демонстрация filter_by_class():")
print("=" * 60)
zebras_only = filter_by_class(df, 'zebra')
print(f"Всего строк: {len(df)}, после фильтра 'zebra': {len(zebras_only)}")
print(zebras_only[['class_label', 'absolute_path']].head(3).to_string(index=False))


# ============================================================
# ЗАДАНИЕ 7. Функция filter_by_class_and_size(df, class_label,
#            max_height, max_width) — фильтрация по метке и размерам.
#            Условие: class_label == x AND height ≤ max_height
#                     AND width ≤ max_width.
# ============================================================
print("\n" + "=" * 60)
print("[Задание 7] Демонстрация filter_by_class_and_size():")
print("=" * 60)
small_horses = filter_by_class_and_size(
    df, class_label='bay_horse', max_height=500, max_width=500,
)
print(f"bay_horse с height≤500 и width≤500: {len(small_horses)} шт.")
print(small_horses[['class_label', 'height', 'width']].head(3).to_string(index=False))


# ============================================================
# ЗАДАНИЕ 8. Группировка по метке класса с вычислением
#            min / max / mean по количеству пикселей.
#            Новый столбец 'pixels' = height * width * depth.
#            + график 3–5 изображений с подписями.
# ============================================================
df['pixels'] = df['height'] * df['width'] * df['depth']

print("\n" + "=" * 60)
print("[Задание 8] Статистика по количеству пикселей на класс:")
print("=" * 60)

pixel_stats = (
    df.groupby('class_label', observed=True)['pixels']
      .agg(
          min_pixels='min',
          max_pixels='max',
          mean_pixels='mean',
          median_pixels='median',
          total_pixels='sum',
          count='size',
      )
      .reset_index()
)
for col in ['min_pixels', 'max_pixels', 'mean_pixels', 'median_pixels', 'total_pixels']:
    pixel_stats[col.replace('pixels', 'Mp')] = (pixel_stats[col] / 1_000_000).round(3)

print(pixel_stats.to_string(index=False))

pivot_pixels = df.pivot_table(
    index='class_label',
    values='pixels',
    aggfunc=['min', 'max', 'mean'],
    observed=True,
)
print("\nPivot (min / max / mean по количеству пикселей):")
print(pivot_pixels)

# --- График 3–5 изображений с подписями ---
N = 5
sample = df.sample(n=min(N, len(df)), random_state=42).reset_index(drop=True)

fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
if N == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, sample.iterrows()):
    img = cv2.imread(row['absolute_path'])
    if img is None:
        ax.text(0.5, 0.5, 'не прочитано', ha='center')
        ax.axis('off')
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(
        f"{row['class_label']}\n"
        f"{row['width']}×{row['height']}×{row['depth']}\n"
        f"{row['pixels'] / 1e6:.2f} Мп",
        fontsize=10,
    )
    ax.axis('off')

plt.suptitle('[Задание 8] Примеры изображений из датасета', fontsize=14)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=120)
plt.show()

# Boxplot количества пикселей по классам
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='pixels', by='class_label', ax=ax, grid=False)
ax.set_title('[Задание 8] Распределение количества пикселей по классам')
ax.set_xlabel('Класс')
ax.set_ylabel('Пиксели')
plt.suptitle('')
plt.tight_layout()
plt.savefig('pixels_by_class.png', dpi=120)
plt.show()

print("\n[Задание 8] ВЫВОД ПО КОЛИЧЕСТВУ ПИКСЕЛЕЙ:")
for _, row in pixel_stats.iterrows():
    print(f"{row['class_label']:>10}: "
          f"min={row['min_Mp']:.3f} Мп, "
          f"max={row['max_Mp']:.3f} Мп, "
          f"mean={row['mean_Mp']:.3f} Мп")

mean_ratio = pixel_stats['mean_pixels'].max() / pixel_stats['mean_pixels'].min()
print(f"Отношение средних пикселей max/min = {mean_ratio:.2f}")
if mean_ratio <= 1.2:
    print("Средние размеры классов близки — изображения сопоставимы.")
else:
    print("Средние размеры классов различаются — "
          "при обучении стоит выполнить resize к единому размеру.")


# ============================================================
# ЗАДАНИЕ 9. Функция get_channel_histograms(df, class_label, ...)
#            — строит гистограммы по каналам B, G, R средствами
#            OpenCV (cv2.calcHist), случайно выбирая изображение
#            нужного класса (numpy rng).
#            Реализована в get_channel_histograms.py.
#            Здесь — демонстрация вызова для обоих классов
#            (результаты используются в ЗАДАНИИ 10).
# ============================================================
CLASSES = ['bay_horse', 'zebra']
COLORS  = {'B': 'blue', 'G': 'green', 'R': 'red'}

# Сохраняем массивы гистограмм по классам для задания 10
hists = {}
for cls in CLASSES:
    print(f"\n[Задание 9] Построение гистограмм для класса '{cls}':")
    hists[cls] = get_channel_histograms(df, cls, bins=256, seed=42)


# ============================================================
# ЗАДАНИЕ 10. Отрисовка гистограмм из задания 9 средствами matplotlib.
#             На графиках — подписи осей, легенда, заголовки.
# ============================================================
fig, axes = plt.subplots(len(CLASSES), 1, figsize=(10, 4 * len(CLASSES)))
if len(CLASSES) == 1:
    axes = [axes]

x = np.arange(256)
for ax, cls in zip(axes, CLASSES):
    hist_b, hist_g, hist_r = hists[cls]

    ax.plot(x, hist_b, color=COLORS['B'], label='Blue (канал 0)')
    ax.plot(x, hist_g, color=COLORS['G'], label='Green (канал 1)')
    ax.plot(x, hist_r, color=COLORS['R'], label='Red (канал 2)')

    ax.set_title(f'[Задание 10] Гистограмма яркости по каналам — класс «{cls}»')
    ax.set_xlabel('Интенсивность пикселя (0–255)')
    ax.set_ylabel('Доля пикселей (нормировано, площадь = 1)')
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.3)
    ax.legend(title='Канал')

plt.tight_layout()
plt.savefig('channel_histograms.png', dpi=120)
plt.show()

print("\n" + "=" * 60)
print("[Задание 9–10] Гистограммы по каналам B, G, R построены.")
print("=" * 60)
print("Гистограммы нормированы на количество пикселей (площадь = 1),")
print("поэтому их можно сравнивать между изображениями разного размера.")