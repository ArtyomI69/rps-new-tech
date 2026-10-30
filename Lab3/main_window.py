from nicegui import ui

from build_annotation import build_annotation
from shuffle_dataset import shuffle_dataset
from zebra import zebra
from bay_horse import bay_horse


class State:
    def __init__(self):
        self.zebra = next(zebra)
        self.bay_horse = next(bay_horse)


state = State()


def handle_next_zebra_btn_click():
    state.zebra = next(zebra)


def handle_next_bay_horse_btn_click():
    state.bay_horse = next(bay_horse)


@ui.page('/')
def index():
    # Кнопка перехода на страницу /dataset-form
    ui.button('Перейти к форме датасета', on_click=lambda: ui.navigate.to('/dataset-form'))

    with ui.row().classes('w-full no-wrap'):
        # Левая половина — зебра
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next zebra', on_click=handle_next_zebra_btn_click)
            ui.image().bind_source_from(state, 'zebra').classes('w-full')

        # Правая половина — bay horse
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next bay horse', on_click=handle_next_bay_horse_btn_click)
            ui.image().bind_source_from(state, 'bay_horse').classes('w-full')

@ui.page('/dataset-form')
def dataset_form():
    def create_annotation():
        build_annotation(annotation_input_dataset.value, annotation_filename_input.value)

    def shuffle_annotations():
        shuffle_dataset(shuffle_input_dataset.value, shuffled_dataset_name_input.value, shuffled_annotation_filename_input.value)

    def go_home():
        ui.navigate.to('/')


    # Кнопка возврата на главную
    ui.button('Вернуться на главную', on_click=go_home).classes('mb-4')

    # Инпут + кнопка создать аннотацию
    with ui.row().classes('w-full items-center'):
        annotation_input_dataset = ui.input(placeholder='Введите путь до датасета (например "dataset")').classes('flex-grow')
        annotation_filename_input = ui.input(placeholder='Введите название файла анотации (например "annotation.csv")').classes('flex-grow')
        ui.button('Создать аннотацию', on_click=create_annotation)

    # Кнопка перемешать
    with ui.row().classes('w-full items-center'):
        shuffle_input_dataset = ui.input(placeholder='Введите путь до датасета (например "dataset")').classes('flex-grow')
        shuffled_dataset_name_input = ui.input(placeholder='Введите название пермешанного датасета (например "dataset_shuffled")').classes('flex-grow')
        shuffled_annotation_filename_input = ui.input(placeholder='Введите название файла анотации перемешанного датасета (например "annotation_shuffled.csv")').classes('flex-grow')
        ui.button('Перемешать', on_click=shuffle_annotations).classes('mb-4')

ui.run()