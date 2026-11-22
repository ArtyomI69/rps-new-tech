import csv
import random
from pathlib import Path

import torch
from nicegui import ui

from zebra import zebra
from bay_horse import bay_horse

# Переиспользуем проверенный код из predict.py
from predict import load_model, build_transform, predict as _predict


# ---------- Пути ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'lr1-e4' / 'best_model.pt'
ANNOTATION_PATH = BASE_DIR / 'annotation.csv'

# ---------- Модель (загружается один раз при старте процесса) ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = load_model(MODEL_PATH, device)
_tf = build_transform(224)


def predict(image_path: str) -> dict:
    """Обёртка над predict.predict с уже загруженной моделью."""
    return _predict(image_path, _model, device, _tf)


# ---------- Датасет ----------
def load_dataset() -> list[tuple[str, str]]:
    """Возвращает список (абсолютный_путь, истинный_класс)."""
    items: list[tuple[str, str]] = []
    with open(ANNOTATION_PATH, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            rel = row['relative_path'].replace('\\', '/')
            items.append((str(BASE_DIR / rel), row['class_label']))
    return items


dataset = load_dataset()


# ---------- Состояние ----------
class State:
    def __init__(self):
        self.zebra = next(zebra)
        self.bay_horse = next(bay_horse)
        self.loop = False


state = State()


# ---------- Обработчики: зебра / лошадь ----------
def handle_next_zebra_btn_click():
    state.zebra = next(zebra)


def handle_next_bay_horse_btn_click():
    state.bay_horse = next(bay_horse)


def handle_loop_change(e):
    state.loop = e.value
    bay_horse.set_loop(e.value)
    zebra.set_loop(e.value)


# ---------- UI: главная страница ----------
@ui.page('/')
def index():
    ui.link('Перейти в Lab6 (predict)', '/lab6').classes('text-lg')

    # Чекбокс зацикливания
    ui.checkbox('Loop', value=state.loop, on_change=handle_loop_change)

    with ui.row().classes('w-full no-wrap'):
        # Левая половина — зебра
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next zebra', on_click=handle_next_zebra_btn_click)
            ui.image().bind_source_from(state, 'zebra').classes('w-full')

        # Правая половина — bay horse
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next bay horse', on_click=handle_next_bay_horse_btn_click)
            ui.image().bind_source_from(state, 'bay_horse').classes('w-full')


# ---------- UI: страница /lab6 ----------
@ui.page('/lab6')
def lab6():
    ui.link('← На главную', '/').classes('text-lg')

    # Локальное состояние страницы
    page_state = {
        'current_image_path': None,
        'current_true_label': None,
    }

    def handle_random_image_btn_click():
        path, label = random.choice(dataset)
        page_state['current_image_path'] = path
        page_state['current_true_label'] = label

        preview.set_source(path)
        true_label_label.set_text(f'True label: {label}')
        prediction_label.set_text('Prediction: —')

    def handle_predict_btn_click():
        if not page_state['current_image_path']:
            ui.notify('Сначала загрузите картинку', type='warning')
            return
        try:
            r = predict(page_state['current_image_path'])
            prediction_label.set_text(
                f"Prediction: {r['predicted_class']} ({r['confidence']:.2%})"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            ui.notify(f'Ошибка: {e}', type='negative')

    with ui.column().classes('w-full items-center'):
        with ui.row().classes('items-center'):
            ui.button('Random image', on_click=handle_random_image_btn_click)
            ui.button('Predict', on_click=handle_predict_btn_click)
            true_label_label = ui.label('True label: —').classes('text-lg')
            prediction_label = ui.label('Prediction: —').classes('text-lg font-bold')

        preview = ui.image().classes('w-1/2')


ui.run(reload=False)