import requests, os;
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, parse_qs, unquote, quote
from utils.pad_filename import pad_filename
import time

REQUEST_URL = 'https://yandex.ru/images/search?text='

def download_yandex_images(text):
    with sync_playwright() as p:
        browser = p.webkit.launch(headless=False)
        page = browser.new_page()
        page.goto(REQUEST_URL+quote(text))
        
        # Ждём загрузки страницы
        show_more_button = page.get_by_text("Показать ещё")
        show_more_button.wait_for(state="visible")

        # Скролл пока не пропадёт "Показать ещё" кнопка
        while show_more_button.is_visible():
            show_more_button.scroll_into_view_if_needed()
            show_more_button.click()
            page.wait_for_timeout(2000)
            page.wait_for_selector(".Span2", state="hidden")
        
        folder = "dataset/" + text
        # Создаём папку
        os.makedirs(folder, exist_ok=True)

        # Получаем все теги <a>, у которых дочерний элемент - изображение (здесь рекламы нет)
        a_tags = page.query_selector_all(".ImagesContentImage-Cover")
        
        for i, a_tag in enumerate(a_tags):
            try:
                # Извлекаем url к полноразмерному изображению из href и затем из queryParam img_url
                href = a_tag.get_attribute("href")
                query = urlparse(href).query
                params = parse_qs(query)
                img_url = unquote(params["img_url"][0])

                # Скачиваем изображение и помещаем в папку dataset/{text}
                filename = os.path.join(folder, pad_filename(i, prefix=text+'_', extension=".jpg"))
                response = requests.get(img_url, timeout=(5, 30))
                response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"[{i}] пропущено: {e}")
                continue

            
            time.sleep(2) 

        browser.close()
