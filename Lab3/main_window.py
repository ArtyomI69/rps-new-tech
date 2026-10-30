from nicegui import ui

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
    with ui.row().classes('w-full no-wrap'):
        # Левая половина — зебра
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next zebra', on_click=handle_next_zebra_btn_click)
            ui.image().bind_source_from(state, 'zebra').classes('w-full')

        # Правая половина — bay horse
        with ui.column().classes('w-1/2 items-center'):
            ui.button('Next bay horse', on_click=handle_next_bay_horse_btn_click)
            ui.image().bind_source_from(state, 'bay_horse').classes('w-full')


ui.run()