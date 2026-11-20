# Используем официальный образ Python (slim-версия для уменьшения размера)
FROM python:3.12-slim

# Устанавливаем переменные окружения
# PYTHONDONTWRITEBYTECODE — не создавать .pyc файлы
# PYTHONUNBUFFERED — логи выводятся сразу (важно для Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости
# build-essential и libgl нужны для opencv-python (cv2)
# libglib2.0-0 и libsm6, libxext6, libxrender1 — для работы OpenCV/Matplotlib в headless-режиме
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt — для кэширования слоя с зависимостями
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Порт, который слушает приложение (NiceGUI по умолчанию 8080)
EXPOSE 8080

# Запуск приложения
CMD ["python", "main_window.py"]