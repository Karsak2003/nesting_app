"""
Точка входа клиентского приложения ИАГИ.
Инициализирует приложение, применяет тему и запускает главный интерфейс.
"""
#region Imports
import sys

#from PyQt5.QtWidgets import QApplication
from PySide6.QtWidgets import QApplication
 
# Импорт из нашего нового структурированного пакета
#from gui import MainWindow, get_client_config
from gui.main_window import MainWindow
from gui.config import get_client_config

# Импорт модуля тем (предполагается, что themes.py лежит в той же директории 
# или в пакете iagi_client. Если в пакете, измените на: from iagi_client.themes import ...)
from gui.themes import get_palette, generate_stylesheet
#endregion

def main():
    # 1. Инициализация приложения
    app = QApplication(sys.argv)
    
    # Устанавливаем базовый стиль Fusion, который лучше всего работает с кастомными QSS
    app.setStyle('Fusion')
    
    # 2. Загрузка конфигурации
    config = get_client_config()
    theme_name = config.get("theme", "system")
    server_url = config.get("server", {}).get("url", "http://localhost:8000")
    
    # 3. Применение темы через модуль themes.py
    # Получаем объект палитры на основе настроек (или системной темы)
    palette = get_palette(theme_name)
    
    # Генерируем строку QSS стилей
    stylesheet = generate_stylesheet(palette)
    
    # Применяем стили ко всему приложению глобально
    app.setStyleSheet(stylesheet)
    
    # 4. Создание и отображение главного окна
    # Передаем начальные параметры, окно само подключится к API и настроит интерфейс
    window = MainWindow(server_url=server_url, theme=theme_name)
    window.show()
    
    # 5. Запуск цикла событий
    sys.exit(app.exec())

if __name__ == "__main__": main()