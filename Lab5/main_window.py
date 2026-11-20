import asyncio
import base64
import io
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from nicegui import app, ui

from build_annotation import build_annotation
from shuffle_dataset import shuffle_dataset
from zebra import zebra
from bay_horse import bay_horse

from get_image_info import get_image_info
from get_channel_histograms import get_channel_histograms
from filter_by_class import filter_by_class, filter_by_class_and_size


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_PATH = PROJECT_ROOT / 'annotation.csv'


# --------------------------------------------------------------------------
#  Утилиты конвертации
# --------------------------------------------------------------------------
def fig_to_data_url(fig, dpi=110):
    """matplotlib Figure -> PNG base64 data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')


def cv2_to_data_url(img_bgr, max_width=400):
    """OpenCV-изображение -> PNG base64 data URL (с ужатием до max_width)."""
    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        img_bgr = cv2.resize(img_bgr, (max_width, int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')


# --------------------------------------------------------------------------
#  Состояние главной страницы
# --------------------------------------------------------------------------
class State:
    def __init__(self):
        self.zebra = next(zebra)
        self.bay_horse = next(bay_horse)
        self.loop = False


state = State()


def handle_next_zebra_btn_click():
    state.zebra = next(zebra)


def handle_next_bay_horse_btn_click():
    state.bay_horse = next(bay_horse)


def handle_loop_change(e):
    state.loop = e.value
    bay_horse.set_loop(e.value)
    zebra.set_loop(e.value)


# --------------------------------------------------------------------------
#  Главная
# --------------------------------------------------------------------------
@ui.page('/')
def index():
    ui.button('Перейти к форме датасета',
              on_click=lambda: ui.navigate.to('/dataset-form'))
    ui.button('Lab4', on_click=lambda: ui.navigate.to('/Lab4'))

    ui.checkbox('Loop', value=state.loop, on_change=handle_loop_change)

    with ui.row().classes('w-full no-wrap'):
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next zebra', on_click=handle_next_zebra_btn_click)
            ui.image().bind_source_from(state, 'zebra').classes('w-full')

        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next bay horse', on_click=handle_next_bay_horse_btn_click)
            ui.image().bind_source_from(state, 'bay_horse').classes('w-full')


# --------------------------------------------------------------------------
#  Страница /dataset-form (без изменений)
# --------------------------------------------------------------------------
@ui.page('/dataset-form')
def dataset_form():
    def create_annotation():
        build_annotation(annotation_input_dataset.value,
                         annotation_filename_input.value)

    def shuffle_annotations():
        shuffle_dataset(shuffle_input_dataset.value,
                        shuffled_dataset_name_input.value,
                        shuffled_annotation_filename_input.value)

    def go_home():
        ui.navigate.to('/')

    ui.button('Вернуться на главную', on_click=go_home).classes('mb-4')

    with ui.row().classes('w-full items-center'):
        annotation_input_dataset = ui.input(
            placeholder='Введите путь до датасета (например "dataset")'
        ).classes('flex-grow')
        annotation_filename_input = ui.input(
            placeholder='Введите название файла анотации (например "annotation.csv")'
        ).classes('flex-grow')
        ui.button('Создать аннотацию', on_click=create_annotation)

    with ui.row().classes('w-full items-center'):
        shuffle_input_dataset = ui.input(
            placeholder='Введите путь до датасета (например "dataset")'
        ).classes('flex-grow')
        shuffled_dataset_name_input = ui.input(
            placeholder='Введите название пермешанного датасета (например "dataset_shuffled")'
        ).classes('flex-grow')
        shuffled_annotation_filename_input = ui.input(
            placeholder='Введите название файла анотации перемешанного датасета (например "annotation_shuffled.csv")'
        ).classes('flex-grow')
        ui.button('Перемешать', on_click=shuffle_annotations).classes('mb-4')


# --------------------------------------------------------------------------
#  Страница /Lab4 — 10 отдельных заданий
# --------------------------------------------------------------------------
@ui.page('/Lab4')
def lab4():
    def go_home():
        ui.navigate.to('/')

    ui.button('Вернуться на главную', on_click=go_home).classes('mb-4')
    ui.label('Лабораторная работа 4').classes('text-2xl mb-2')
    ui.label(
        'Каждое задание запускается отдельной кнопкой. '
        'Результаты сохраняются — можно выполнять задания по порядку. '
        'Задания 3–10 зависят от результатов предыдущих.'
    ).classes('text-gray-600 mb-4')

    # Per-client state
    session_state = {'df': None, 'hists': {}}

    # -------- вспомогательные функции --------
    def make_task_card(num, title):
        """Создаёт карточку задания. Возвращает (status_label, button, results_column)."""
        with ui.card().classes('w-full'):
            with ui.row().classes('items-center w-full no-wrap'):
                status = ui.label('').classes('text-2xl w-8 text-center')
                ui.label(f'Задание {num}. {title}').classes('text-lg font-bold flex-grow')
                btn = ui.button('Выполнить', color='primary')
            results = ui.column().classes('w-full gap-2')
        return status, btn, results

    def show_warning(results, message):
        results.clear()
        with results:
            ui.label(message).classes('text-orange-600 font-bold')

    def df_ready(results, need_cols=()):
        """Проверяет готовность df и нужных колонок; пишет предупреждение при необходимости."""
        df = session_state['df']
        if df is None:
            show_warning(results, 'Сначала выполните Задание 1 и Задание 2')
            return False
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            show_warning(
                results,
                f'Не хватает колонок {missing}. Сначала выполните '
                f'задания, которые их добавляют.'
            )
            return False
        return True

    # =====================================================
    #  Задание 1
    # =====================================================
    status1, btn1, results1 = make_task_card(1, 'DataFrame из аннотации')

    async def run_task_1(*_):
        results1.clear()
        if not ANNOTATION_PATH.exists():
            with results1:
                ui.label(f'Файл не найден: {ANNOTATION_PATH}').classes('text-red-600')
            status1.text = '❌'
            return
        try:
            df = pd.read_csv(ANNOTATION_PATH, sep=';', encoding='utf-8-sig')
            session_state['df'] = df
            with results1:
                ui.label('Исходные колонки: ' + ', '.join(df.columns))
                ui.label(f'Всего строк: {len(df)}')
                ui.label('Первые 5 строк:').classes('font-bold')
                ui.table.from_pandas(df.head(5))
            status1.text = '✅'
        except Exception as e:
            with results1:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status1.text = '❌'

    btn1.on('click', run_task_1)

    # =====================================================
    #  Задание 2
    # =====================================================
    status2, btn2, results2 = make_task_card(2, 'Нормализация имён колонок')

    async def run_task_2(*_):
        if session_state['df'] is None:
            show_warning(results2, 'Сначала выполните Задание 1')
            status2.text = '⚠️'
            return
        results2.clear()
        try:
            df = session_state['df']
            before = list(df.columns)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            after = list(df.columns)
            df = df[['class_label', 'absolute_path']]
            session_state['df'] = df
            with results2:
                ui.label('До нормализации: ' + ', '.join(before))
                ui.label('После нормализации: ' + ', '.join(after))
                ui.label(
                    'Оставлены 2 колонки: class_label, absolute_path'
                ).classes('font-bold mt-2')
                ui.table.from_pandas(df.head(5))
            status2.text = '✅'
        except Exception as e:
            with results2:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status2.text = '❌'

    btn2.on('click', run_task_2)

    # =====================================================
    #  Задание 3
    # =====================================================
    status3, btn3, results3 = make_task_card(3, 'Числовые метки классов')

    async def run_task_3(*_):
        if not df_ready(results3, ['class_label']):
            status3.text = '⚠️'
            return
        try:
            df = session_state['df']
            df['class_id'] = pd.Categorical(
                df['class_label'], categories=['bay_horse', 'zebra']
            ).codes
            session_state['df'] = df
            results3.clear()
            with results3:
                ui.table.from_pandas(
                    df[['class_label', 'class_id']].drop_duplicates()
                )
            status3.text = '✅'
        except Exception as e:
            with results3:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status3.text = '❌'

    btn3.on('click', run_task_3)

    # =====================================================
    #  Задание 4
    # =====================================================
    status4, btn4, results4 = make_task_card(
        4, 'Размеры изображений (height, width, depth)'
    )

    async def run_task_4(*_):
        if not df_ready(results4, ['absolute_path']):
            status4.text = '⚠️'
            return
        results4.clear()
        with results4:
            loading = ui.label('Чтение размеров изображений... Это может занять время.')\
                .classes('text-gray-600')
        await asyncio.sleep(0.05)   # дать UI отрисовать сообщение
        try:
            df = session_state['df']
            info = df['absolute_path'].apply(get_image_info)
            df[['height', 'width', 'depth']] = pd.DataFrame(
                info.tolist(), index=df.index
            )
            session_state['df'] = df
            results4.clear()
            with results4:
                ui.label(f'Размеры прочитаны для {len(df)} изображений.')
                ui.table.from_pandas(
                    df[['class_label', 'height', 'width', 'depth']].head()
                )
            status4.text = '✅'
        except Exception as e:
            results4.clear()
            with results4:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status4.text = '❌'

    btn4.on('click', run_task_4)

    # =====================================================
    #  Задание 5
    # =====================================================
    status5, btn5, results5 = make_task_card(
        5, 'Статистика размеров и баланс классов'
    )

    async def run_task_5(*_):
        if not df_ready(results5, ['height', 'width', 'depth']):
            status5.text = '⚠️'
            return
        results5.clear()
        try:
            df = session_state['df']
            with results5:
                size_stats = df[['height', 'width', 'depth']].describe().reset_index()
                ui.label('describe():').classes('font-bold')
                ui.table.from_pandas(size_stats)

                extra = pd.DataFrame({
                    'median': df[['height', 'width', 'depth']].median(),
                    'mode':   df[['height', 'width', 'depth']].mode().iloc[0],
                    'range':  df[['height', 'width', 'depth']].max()
                            - df[['height', 'width', 'depth']].min(),
                    'IQR':    df[['height', 'width', 'depth']].quantile(0.75)
                            - df[['height', 'width', 'depth']].quantile(0.25),
                }).reset_index()
                ui.label('Дополнительные характеристики:').classes('font-bold mt-4')
                ui.table.from_pandas(extra)

                label_summary = (
                    df.groupby('class_label', observed=True)
                      .agg(
                          count=('class_label', 'size'),
                          height_mean=('height', 'mean'),
                          width_mean=('width', 'mean'),
                      )
                      .reset_index()
                )
                label_summary['share_%'] = (
                    label_summary['count'] / label_summary['count'].sum() * 100
                ).round(2)
                ui.label('Сводка по классам:').classes('font-bold mt-4')
                ui.table.from_pandas(label_summary)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                label_summary.plot(
                    kind='bar', x='class_label', y='count',
                    ax=axes[0], color=['#4C72B0', '#DD8452'], legend=False,
                )
                axes[0].set_title('Количество изображений по классам')
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
                axes[1].set_title('Доли классов')
                plt.tight_layout()
                ui.image(fig_to_data_url(fig)).classes('w-full mt-4')

                counts = label_summary.set_index('class_label')['count']
                ratio = (counts.max() / counts.min()
                         if counts.min() > 0 else float('inf'))
                if ratio <= 1.2:
                    verdict = 'СБАЛАНСИРОВАН (различие ≤ 20%)'
                elif ratio <= 1.5:
                    verdict = 'УМЕРЕННО НЕСБАЛАНСИРОВАН (20–50%)'
                else:
                    verdict = 'НЕСБАЛАНСИРОВАН (> 50%)'
                ui.label(
                    f'Количество в классах: {counts.to_dict()}\n'
                    f'Соотношение max/min = {ratio:.2f}\n'
                    f'Итог: {verdict}'
                ).classes('font-bold mt-4 whitespace-pre-line')
            status5.text = '✅'
        except Exception as e:
            with results5:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status5.text = '❌'

    btn5.on('click', run_task_5)

    # =====================================================
    #  Задание 6
    # =====================================================
    status6, btn6, results6 = make_task_card(6, 'filter_by_class')

    async def run_task_6(*_):
        if not df_ready(results6, ['class_label']):
            status6.text = '⚠️'
            return
        results6.clear()
        try:
            df = session_state['df']
            zebras_only = filter_by_class(df, 'zebra')
            with results6:
                ui.label(
                    f'Всего строк: {len(df)}, '
                    f'после фильтра "zebra": {len(zebras_only)}'
                )
                ui.table.from_pandas(
                    zebras_only[['class_label', 'absolute_path']].head(3)
                )
            status6.text = '✅'
        except Exception as e:
            with results6:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status6.text = '❌'

    btn6.on('click', run_task_6)

    # =====================================================
    #  Задание 7
    # =====================================================
    status7, btn7, results7 = make_task_card(7, 'filter_by_class_and_size')

    async def run_task_7(*_):
        if not df_ready(results7, ['class_label', 'height', 'width']):
            status7.text = '⚠️'
            return
        results7.clear()
        try:
            df = session_state['df']
            small_horses = filter_by_class_and_size(
                df, class_label='bay_horse', max_height=500, max_width=500,
            )
            with results7:
                ui.label(
                    f'bay_horse с height≤500 и width≤500: '
                    f'{len(small_horses)} шт.'
                )
                ui.table.from_pandas(
                    small_horses[['class_label', 'height', 'width']].head(3)
                )
            status7.text = '✅'
        except Exception as e:
            with results7:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status7.text = '❌'

    btn7.on('click', run_task_7)

    # =====================================================
    #  Задание 8
    # =====================================================
    status8, btn8, results8 = make_task_card(
        8, 'Статистика по пикселям + примеры изображений'
    )

    async def run_task_8(*_):
        if not df_ready(results8, ['height', 'width', 'depth']):
            status8.text = '⚠️'
            return
        results8.clear()
        try:
            df = session_state['df']
            df['pixels'] = df['height'] * df['width'] * df['depth']
            session_state['df'] = df

            with results8:
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
                ui.label('Статистика по пикселям:').classes('font-bold')
                ui.table.from_pandas(pixel_stats)

                pivot_pixels = df.pivot_table(
                    index='class_label',
                    values='pixels',
                    aggfunc=['min', 'max', 'mean'],
                    observed=True,
                )
                pivot_pixels.columns = ['_'.join(map(str, c))
                                        for c in pivot_pixels.columns]
                pivot_pixels = pivot_pixels.reset_index()
                ui.label('Pivot (min / max / mean):').classes('font-bold mt-4')
                ui.table.from_pandas(pivot_pixels)

                N = 5
                sample = df.sample(
                    n=min(N, len(df)), random_state=42
                ).reset_index(drop=True)
                ui.label('Примеры изображений:').classes('font-bold mt-4')
                with ui.row().classes('w-full items-start flex-wrap'):
                    for _, row in sample.iterrows():
                        img = cv2.imread(row['absolute_path'])
                        with ui.card().classes('p-2'):
                            if img is None:
                                ui.label('не прочитано')
                            else:
                                ui.image(cv2_to_data_url(img)).classes('w-48')
                                ui.label(
                                    f"{row['class_label']}\n"
                                    f"{row['width']}×{row['height']}×{row['depth']}\n"
                                    f"{row['pixels'] / 1e6:.2f} Мп"
                                ).classes(
                                    'text-sm text-center whitespace-pre-line'
                                )

                fig, ax = plt.subplots(figsize=(8, 5))
                df.boxplot(column='pixels', by='class_label', ax=ax, grid=False)
                ax.set_title('Распределение количества пикселей по классам')
                ax.set_xlabel('Класс')
                ax.set_ylabel('Пиксели')
                plt.suptitle('')
                plt.tight_layout()
                ui.image(fig_to_data_url(fig)).classes('w-full mt-4')

                ui.label('Вывод по количеству пикселей:').classes('font-bold mt-4')
                for _, row in pixel_stats.iterrows():
                    ui.label(
                        f"{row['class_label']}: "
                        f"min={row['min_pixels'] / 1e6:.3f} Мп, "
                        f"max={row['max_pixels'] / 1e6:.3f} Мп, "
                        f"mean={row['mean_pixels'] / 1e6:.3f} Мп"
                    )
                mean_ratio = (pixel_stats['mean_pixels'].max()
                              / pixel_stats['mean_pixels'].min())
                if mean_ratio <= 1.2:
                    msg = ('Средние размеры классов близки — '
                           'изображения сопоставимы.')
                else:
                    msg = ('Средние размеры классов различаются — '
                           'при обучении стоит выполнить resize к единому размеру.')
                ui.label(f'Отношение средних max/min = {mean_ratio:.2f}. {msg}')
            status8.text = '✅'
        except Exception as e:
            with results8:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status8.text = '❌'

    btn8.on('click', run_task_8)

    # =====================================================
    #  Задание 9
    # =====================================================
    status9, btn9, results9 = make_task_card(
        9, 'Гистограммы по каналам (cv2.calcHist)'
    )

    async def run_task_9(*_):
        if not df_ready(results9, ['class_label', 'absolute_path']):
            status9.text = '⚠️'
            return
        results9.clear()
        try:
            df = session_state['df']
            CLASSES = ['bay_horse', 'zebra']
            hists = {}
            with results9:
                for cls in CLASSES:
                    try:
                        hists[cls] = get_channel_histograms(
                            df, cls, bins=256, seed=42
                        )
                        ui.label(f'✅ Гистограммы для «{cls}» построены')
                    except Exception as e:
                        ui.label(f'❌ {cls}: {e}').classes('text-red-600')
            session_state['hists'] = hists
            status9.text = '✅' if len(hists) == len(CLASSES) else '⚠️'
        except Exception as e:
            with results9:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status9.text = '❌'

    btn9.on('click', run_task_9)

    # =====================================================
    #  Задание 10
    # =====================================================
    status10, btn10, results10 = make_task_card(
        10, 'Отрисовка гистограмм (matplotlib)'
    )

    async def run_task_10(*_):
        hists = session_state['hists']
        if not hists:
            show_warning(results10, 'Сначала выполните Задание 9')
            status10.text = '⚠️'
            return
        results10.clear()
        try:
            CLASSES = ['bay_horse', 'zebra']
            COLORS = {'B': 'blue', 'G': 'green', 'R': 'red'}
            with results10:
                fig, axes = plt.subplots(
                    len(CLASSES), 1, figsize=(10, 4 * len(CLASSES))
                )
                if len(CLASSES) == 1:
                    axes = [axes]
                x = np.arange(256)
                for ax, cls in zip(axes, CLASSES):
                    if cls not in hists:
                        ax.set_title(f'Нет данных для «{cls}»')
                        continue
                    hist_b, hist_g, hist_r = hists[cls]
                    ax.plot(x, hist_b, color=COLORS['B'], label='Blue (канал 0)')
                    ax.plot(x, hist_g, color=COLORS['G'], label='Green (канал 1)')
                    ax.plot(x, hist_r, color=COLORS['R'], label='Red (канал 2)')
                    ax.set_title(f'Гистограмма яркости по каналам — класс «{cls}»')
                    ax.set_xlabel('Интенсивность пикселя (0–255)')
                    ax.set_ylabel('Доля пикселей (нормировано, площадь = 1)')
                    ax.set_xlim(0, 255)
                    ax.grid(alpha=0.3)
                    ax.legend(title='Канал')
                plt.tight_layout()
                ui.image(fig_to_data_url(fig)).classes('w-full mt-4')
            status10.text = '✅'
        except Exception as e:
            with results10:
                ui.label(f'Ошибка: {e}').classes('text-red-600')
            status10.text = '❌'

    btn10.on('click', run_task_10)

app.add_static_files('/dataset', os.path.join(os.path.dirname(__file__), 'dataset'))

ui.run()