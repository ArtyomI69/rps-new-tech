import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from get_image_info import get_image_info
from get_channel_histograms import get_channel_histograms
from filter_by_class import filter_by_class, filter_by_class_and_size

# ---- Чтение и подготовка (как в исходнике) ----
df = pd.read_csv('annotation.csv', sep=';', encoding='utf-8-sig')
df = df[['class_label', 'absolute_path']]

df['class_id'] = pd.Categorical(
    df['class_label'],
    categories=['bay horse', 'zebra']
).codes


info = df['absolute_path'].apply(get_image_info)
df[['height', 'width', 'depth']] = pd.DataFrame(info.tolist(), index=df.index)


# ============================================================
# 1. Describe по размерам изображений
# ============================================================
print("=" * 60)
print("Статистика по размерам изображений (describe):")
print("=" * 60)
size_stats = df[['height', 'width', 'depth']].describe()
print(size_stats)

# Дополнительно: медиана, мода, размах
extra = pd.DataFrame({
    'median': df[['height', 'width', 'depth']].median(),
    'mode':   df[['height', 'width', 'depth']].mode().iloc[0],
    'range':  df[['height', 'width', 'depth']].max() - df[['height', 'width', 'depth']].min(),
    'IQR':    df[['height', 'width', 'depth']].quantile(0.75) - df[['height', 'width', 'depth']].quantile(0.25),
})
print("\nДополнительные характеристики:")
print(extra)


# ============================================================
# 2. Сводная таблица по меткам класса (сумма / количество)
# ============================================================
print("\n" + "=" * 60)
print("Сводная таблица по меткам класса:")
print("=" * 60)

label_summary = (
    df.groupby('class_label', observed=True)
      .agg(
          count=('class_label', 'size'),
          sum=('class_id', 'sum'),          # сумма кодов (0 для bay horse)
          height_mean=('height', 'mean'),
          width_mean=('width', 'mean'),
      )
      .reset_index()
)

# Доля каждого класса
label_summary['share_%'] = (
    label_summary['count'] / label_summary['count'].sum() * 100
).round(2)

# pivot_table — «табличка с суммой по метке»
pivot = df.pivot_table(
    index='class_label',
    values=['class_id', 'height', 'width', 'depth'],
    aggfunc=['count', 'sum', 'mean'],
    observed=True
)
print("\nPivot table (count / sum / mean):")
print(pivot)

print("\nСводка по классам:")
print(label_summary.to_string(index=False))


# ============================================================
# 3. Визуализация: гистограмма + pie
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Гистограмма количества по классам
label_summary.plot(
    kind='bar', x='class_label', y='count',
    ax=axes[0], color=['#4C72B0', '#DD8452'], legend=False
)
axes[0].set_title('Количество изображений по классам')
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(label_summary['count']):
    axes[0].text(i, v, str(v), ha='center', va='bottom')

# Pie-диаграмма долей классов
axes[1].pie(
    label_summary['count'],
    labels=label_summary['class_label'],
    autopct='%1.1f%%',
    colors=['#4C72B0', '#DD8452'],
    startangle=90,
    explode=[0.03] * len(label_summary)
)
axes[1].set_title('Доли классов')

plt.tight_layout()
plt.savefig('class_balance.png', dpi=120)
plt.show()


# ============================================================
# 4. Вывод о сбалансированности
# ============================================================
counts = label_summary.set_index('class_label')['count']
ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

print("\n" + "=" * 60)
print("ВЫВОД О СБАЛАНСИРОВАННОСТИ")
print("=" * 60)
print(f"Количество в классах:\n{counts.to_string()}")
print(f"Соотношение max/min = {ratio:.2f}")

if ratio <= 1.2:
    verdict = "СБАЛАНСИРОВАН (различие ≤ 20%)"
elif ratio <= 1.5:
    verdict = "УМЕРЕННО НЕСБАЛАНСИРОВАН (20–50%)"
else:
    verdict = "НЕСБАЛАНСИРОВАН (> 50%)"

print(f"\nИтог: {verdict}")

# ============================================================
# 5. Количество пикселей: новый столбец + группировка
# ============================================================
# pixels = height * width * depth (каналы)
df['pixels'] = df['height'] * df['width'] * df['depth']

print("\n" + "=" * 60)
print("Количество пикселей: статистика по классам")
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

# «Красивое» округление — миллионы пикселей (Мп)
for col in ['min_pixels', 'max_pixels', 'mean_pixels', 'median_pixels', 'total_pixels']:
    pixel_stats[col.replace('pixels', 'Mp')] = (pixel_stats[col] / 1_000_000).round(3)

print(pixel_stats.to_string(index=False))

# Pivot-таблица: min / max / mean по классам
pivot_pixels = df.pivot_table(
    index='class_label',
    values='pixels',
    aggfunc=['min', 'max', 'mean'],
    observed=True
)
print("\nPivot: min / max / mean количества пикселей:")
print(pivot_pixels)

# ============================================================
# 6. График 3–5 изображений с подписями
# ============================================================
N = 5  # 3–5 изображений
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
        f"{row['pixels']/1e6:.2f} Мп",
        fontsize=10
    )
    ax.axis('off')

plt.suptitle('Примеры изображений из датасета', fontsize=14)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=120)
plt.show()

# ============================================================
# 7. Boxplot количества пикселей по классам
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='pixels', by='class_label', ax=ax, grid=False)
ax.set_title('Распределение количества пикселей по классам')
ax.set_xlabel('Класс')
ax.set_ylabel('Пиксели')
plt.suptitle('')  # убрать авто-заголовок pandas
plt.tight_layout()
plt.savefig('pixels_by_class.png', dpi=120)
plt.show()

# ============================================================
# 8. Вывод: сравнение пикселей между классами
# ============================================================
print("\n" + "=" * 60)
print("ВЫВОД ПО КОЛИЧЕСТВУ ПИКСЕЛЕЙ")
print("=" * 60)

for _, row in pixel_stats.iterrows():
    print(f"{row['class_label']:>10}: "
          f"min={row['min_Mp']:.3f} Мп, "
          f"max={row['max_Mp']:.3f} Мп, "
          f"mean={row['mean_Mp']:.3f} Мп")

mean_ratio = pixel_stats['mean_pixels'].max() / pixel_stats['mean_pixels'].min()
print(f"\nОтношение средних пикселей max/min = {mean_ratio:.2f}")

if mean_ratio <= 1.2:
    print("Средние размеры классов близки — изображения сопоставимы.")
else:
    print("Средние размеры классов различаются — "
          "при обучении стоит выполнить resize к единому размеру.")

# ============================================================
# 9. Гистограммы по каналам для случайных изображений классов
# ============================================================
CLASSES = ['bay horse', 'zebra']
COLORS  = {'B': 'blue', 'G': 'green', 'R': 'red'}

fig, axes = plt.subplots(len(CLASSES), 1, figsize=(10, 4 * len(CLASSES)))
if len(CLASSES) == 1:
    axes = [axes]

for ax, cls in zip(axes, CLASSES):
    hist_b, hist_g, hist_r = get_channel_histograms(df, cls, bins=256, seed=42)

    x = np.arange(256)
    ax.plot(x, hist_b, color=COLORS['B'], label='Blue (канал 0)')
    ax.plot(x, hist_g, color=COLORS['G'], label='Green (канал 1)')
    ax.plot(x, hist_r, color=COLORS['R'], label='Red (канал 2)')

    ax.set_title(f'Гистограмма яркости по каналам — класс «{cls}»')
    ax.set_xlabel('Интенсивность пикселя (0–255)')
    ax.set_ylabel('Доля пикселей (нормировано)')
    ax.set_xlim(0, 255)
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('channel_histograms.png', dpi=120)
plt.show()


# ============================================================
# 10. Вывод по гистограммам
# ============================================================
print("\n" + "=" * 60)
print("ГИСТОГРАММЫ ЯРКОСТИ ПО КАНАЛАМ")
print("=" * 60)
print("Гистограммы нормированы на количество пикселей (площадь = 1),")
print("поэтому их можно сравнивать между изображениями разного размера.")