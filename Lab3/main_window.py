# main_window.py
import asyncio
from pathlib import Path

from nicegui import ui, run

from dataset_utils import build_annotation, shuffle_dataset, DatasetIterator, get_classes

# ---------- вспомогательные функции ----------

def _pick_folder_dialog(initial: str = "") -> str:
    """Открывает нативный диалог выбора папки (tkinter). Возвращает путь или ''."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(initialdir=initial or None)
        root.destroy()
        return path or ""
    except Exception as e:  # noqa: BLE001
        ui.notify(f"Не удалось открыть диалог: {e}", type="warning")
        return ""


def _pick_save_file_dialog(initial: str = "", default_name: str = "annotation.csv") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            initialdir=initial or None,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return path or ""
    except Exception as e:  # noqa: BLE001
        ui.notify(f"Не удалось открыть диалог: {e}", type="warning")
        return ""


def _summary_text(df) -> str:
    lines = [f"Всего файлов: {len(df)}", "Распределение по классам:"]
    for label, count in df["class_label"].value_counts().sort_index().items():
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)


# ---------- GUI ----------

def create_ui():
    state = {
        "dataset_dir": "",
        "annotation_file": "",
        "shuffled_dir": "",
        "shuffled_annotation": "",
        "iterator": None,
    }

    ui.page_title("Lab2 — Аннотация и итератор датасета")
    ui.label("Лабораторная работа 2: аннотация и возврат изображений").classes(
        "text-2xl font-bold mb-4"
    )

    with ui.tabs().classes("w-full") as tabs:
        tab_ann = ui.tab("Аннотация")
        tab_shuffle = ui.tab("Перемешивание")
        tab_iter = ui.tab("Итератор")

    with ui.tab_panels(tabs, value=tab_ann).classes("w-full"):

        # ---------- вкладка «Аннотация» ----------
        with ui.tab_panel(tab_ann):
            ui.label("1. Укажите папку исходного датасета").classes("text-lg font-semibold")

            with ui.row().classes("w-full items-center"):
                ds_input = ui.input("Путь к папке датасета").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: ds_input.set_value(
                        _pick_folder_dialog(ds_input.value) or ds_input.value
                    ),
                )

            ui.label("2. Куда сохранить файл аннотации (CSV)").classes(
                "text-lg font-semibold mt-4"
            )
            with ui.row().classes("w-full items-center"):
                ann_input = ui.input("Путь к файлу аннотации").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: ann_input.set_value(
                        _pick_save_file_dialog(
                            str(Path(ann_input.value).parent) if ann_input.value else "",
                            "annotation.csv",
                        )
                        or ann_input.value
                    ),
                )

            result_ann = ui.markdown("").classes("mt-4 font-mono whitespace-pre-wrap")

            async def on_build_annotation():
                ds = ds_input.value.strip()
                ann = ann_input.value.strip()
                if not ds or not ann:
                    ui.notify("Укажите папку датасета и файл аннотации", type="warning")
                    return
                try:
                    df = await run.io_bound(
                        build_annotation, ds, ann, project_root=Path(ann).resolve().parent
                    )
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Ошибка: {e}", type="negative", timeout=8000)
                    result_ann.set_content(f"**Ошибка:** {e}")
                    return
                state["dataset_dir"] = ds
                state["annotation_file"] = ann
                result_ann.set_content(
                    f"**Готово.**\n\n```\n{_summary_text(df)}\n```"
                )
                ui.notify("Аннотация создана", type="positive")

            ui.button("Создать аннотацию", on_click=on_build_annotation).classes(
                "mt-4 bg-primary"
            )

        # ---------- вкладка «Перемешивание» ----------
        with ui.tab_panel(tab_shuffle):
            ui.label(
                "Переименовать файлы числовыми именами и создать новую аннотацию"
            ).classes("text-lg font-semibold")

            with ui.row().classes("w-full items-center mt-2"):
                ds2_input = ui.input("Папка исходного датасета").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: ds2_input.set_value(
                        _pick_folder_dialog(ds2_input.value) or ds2_input.value
                    ),
                )

            with ui.row().classes("w-full items-center mt-2"):
                out_dir_input = ui.input("Папка назначения (shuffled)").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: out_dir_input.set_value(
                        _pick_folder_dialog(out_dir_input.value) or out_dir_input.value
                    ),
                )

            with ui.row().classes("w-full items-center mt-2"):
                out_ann_input = ui.input("Файл аннотации (CSV)").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: out_ann_input.set_value(
                        _pick_save_file_dialog(
                            str(Path(out_ann_input.value).parent) if out_ann_input.value else "",
                            "annotation_shuffled.csv",
                        )
                        or out_ann_input.value
                    ),
                )

            with ui.row().classes("items-center mt-2"):
                num_lo = ui.number("Мин. номер", value=0, format="%d").classes("w-32")
                num_hi = ui.number("Макс. номер", value=2000, format="%d").classes("w-32")
                seed_input = ui.number("Seed", value=42, format="%d").classes("w-32")
                copy_switch = ui.switch("Копировать (иначе — перемещать)", value=True)

            result_shuf = ui.markdown("").classes("mt-4 font-mono whitespace-pre-wrap")

            async def on_shuffle():
                ds = ds2_input.value.strip()
                out_dir = out_dir_input.value.strip()
                ann = out_ann_input.value.strip()
                if not ds or not out_dir or not ann:
                    ui.notify("Заполните все три пути", type="warning")
                    return
                try:
                    lo = int(num_lo.value)
                    hi = int(num_hi.value)
                    seed = int(seed_input.value) if seed_input.value is not None else 42
                    df = await run.io_bound(
                        shuffle_dataset,
                        ds,
                        out_dir,
                        ann,
                        num_range=(lo, hi),
                        seed=seed,
                        copy_files=bool(copy_switch.value),
                        project_root=Path(ann).resolve().parent,
                    )
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Ошибка: {e}", type="negative", timeout=8000)
                    result_shuf.set_content(f"**Ошибка:** {e}")
                    return
                state["shuffled_dir"] = out_dir
                state["shuffled_annotation"] = ann
                result_shuf.set_content(
                    f"**Готово.**\n\n```\n{_summary_text(df)}\n```"
                )
                ui.notify("Датасет перемешан", type="positive")

            ui.button("Создать перемешанный датасет", on_click=on_shuffle).classes(
                "mt-4 bg-primary"
            )

        # ---------- вкладка «Итератор» ----------
        with ui.tab_panel(tab_iter):
            ui.label(
                "Итератор по изображениям выбранного класса"
            ).classes("text-lg font-semibold")

            with ui.row().classes("w-full items-center mt-2"):
                ann_iter_input = ui.input("Путь к CSV-аннотации").classes("flex-grow")
                ui.button(
                    "Обзор…",
                    on_click=lambda: ann_iter_input.set_value(
                        _pick_save_file_dialog(
                            str(Path(ann_iter_input.value).parent) if ann_iter_input.value else "",
                            "annotation.csv",
                        )
                        or ann_iter_input.value
                    ),
                )

            class_select = ui.select([], label="Класс").classes("w-64 mt-2")
            iter_log = ui.log(max_lines=200).classes("w-full h-64 mt-2 font-mono")

            def refresh_classes():
                path = ann_iter_input.value.strip()
                if not path:
                    ui.notify("Укажите путь к аннотации", type="warning")
                    return
                try:
                    classes = get_classes(path)
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Ошибка чтения: {e}", type="negative")
                    return
                class_select.options = classes
                if classes:
                    class_select.value = classes[0]
                iter_log.push(f"[{len(classes)}] классов: {classes}")
                state["iterator"] = None

            ui.button("Загрузить классы", on_click=refresh_classes).classes("mt-2")

            def make_iterator():
                path = ann_iter_input.value.strip()
                cls = class_select.value
                if not path or not cls:
                    ui.notify("Укажите аннотацию и класс", type="warning")
                    return
                try:
                    it = DatasetIterator(
                        cls, path, project_root=Path(path).resolve().parent
                    )
                except Exception as e:  # noqa: BLE001
                    ui.notify(f"Ошибка: {e}", type="negative")
                    return
                state["iterator"] = it
                iter_log.push(f"Итератор создан: класс='{cls}', всего={len(it)}")

            def next_item():
                it = state["iterator"]
                if it is None:
                    ui.notify("Сначала создайте итератор", type="warning")
                    return
                try:
                    path = next(it)
                except StopIteration:
                    iter_log.push("StopIteration — элементы закончились")
                    return
                iter_log.push(f"next() -> {path}")

            def reset_iter():
                it = state["iterator"]
                if it is None:
                    return
                it.reset()
                iter_log.push("Итератор сброшен (reset)")

            with ui.row().classes("mt-2"):
                ui.button("Создать итератор", on_click=make_iterator).classes("bg-primary")
                ui.button("next()", on_click=next_item).classes("bg-secondary")
                ui.button("reset()", on_click=reset_iter)


# точка входа
ui.run(title="Lab2 — NiceGUI", reload=False, port=8080)