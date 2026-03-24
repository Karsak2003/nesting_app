"""
Единая точка входа для запуска клиентской или серверной части ИАГИ
Или обеих частей одновременно через парсер аргументов
"""

import sys
import argparse
from pathlib import Path


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Запуск серверной части FastAPI"""
    print(f"🚀 Запуск сервера ИАГИ на {host}:{port}")
    
    try:
        import uvicorn
        from server.main import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        print("❌ Ошибка: uvicorn не установлен. Выполните: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        sys.exit(1)


def run_client(server_url: str = None, theme: str = "system"):
    """Запуск клиентской части PyQt5"""
    print(f"🖥️ Запуск клиента ИАГИ")
    if server_url:
        print(f"   Подключение к серверу: {server_url}")
    else:
        print("   Использование настроек по умолчанию")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from client.gui.interface import MainWindow
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        window = MainWindow(server_url=server_url, theme=theme)
        window.show()
        
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"❌ Ошибка: Не установлены зависимости PyQt5. Выполните: pip install PyQt5 matplotlib")
        print(f"   Детали: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка запуска клиента: {e}")
        sys.exit(1)


def run_both(host: str = "0.0.0.0", port: int = 8000, theme: str = "system"):
    """Запуск обеих частей: сервер в фоне и клиент"""
    import threading
    import time
    
    print(f"🚀 Запуск сервера и клиента ИАГИ")
    
    # Запуск сервера в отдельном потоке
    def start_server():
        try:
            import uvicorn
            from server.main import app
            
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="warning"  # Скрываем логи сервера для чистоты вывода
            )
        except Exception as e:
            print(f"❌ Ошибка сервера: {e}")
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Ждём пока сервер запустится
    print("   ⏳ Ожидание запуска сервера...")
    time.sleep(2)
    
    # Запуск клиента
    server_url = f"http://localhost:{port}"
    run_client(server_url=server_url, theme=theme)


def main():
    parser = argparse.ArgumentParser(
        description="ИАГИ - Система оптимизации раскроя плоских деталей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --server              Запустить только сервер
  python main.py --client              Запустить только клиент
  python main.py --both                Запустить сервер и клиент вместе
  python main.py --server --host 0.0.0.0 --port 8080  Сервер на порту 8080
  python main.py --client --url http://192.168.1.100:8000  Клиент с удалённым сервером
  python main.py --both --theme dark   Запуск всего с тёмной темой
        """
    )
    
    # Режимы запуска
    mode_group = parser.add_argument_group('Режим запуска')
    mode_group.add_argument('--server', action='store_true',
                           help='Запустить только серверную часть (FastAPI)')
    mode_group.add_argument('--client', action='store_true',
                           help='Запустить только клиентскую часть (PyQt5)')
    mode_group.add_argument('--both', action='store_true',
                           help='Запустить обе части одновременно')
    
    # Настройки сервера
    server_group = parser.add_argument_group('Настройки сервера')
    server_group.add_argument('--host', type=str, default='0.0.0.0',
                             help='Хост для сервера (по умолчанию: 0.0.0.0)')
    server_group.add_argument('--port', type=int, default=8000,
                             help='Порт для сервера (по умолчанию: 8000)')
    server_group.add_argument('--reload', action='store_true',
                             help='Включить автоперезагрузку сервера (для разработки)')
    
    # Настройки клиента
    client_group = parser.add_argument_group('Настройки клиента')
    client_group.add_argument('--url', type=str, default=None,
                             help='URL сервера для клиента (по умолчанию: http://localhost:8000)')
    client_group.add_argument('--theme', type=str, choices=['system', 'light', 'dark'],
                             default='system',
                             help='Цветовая тема интерфейса (по умолчанию: system)')
    
    args = parser.parse_args()
    
    # Если ни один режим не указан, показываем справку
    if not (args.server or args.client or args.both):
        parser.print_help()
        print("\n❌ Укажите режим запуска: --server, --client или --both")
        sys.exit(1)
    
    # Запуск в соответствии с выбранным режимом
    if args.both:
        run_both(host=args.host, port=args.port, theme=args.theme)
    elif args.server:
        run_server(host=args.host, port=args.port, reload=args.reload)
    elif args.client:
        run_client(server_url=args.url, theme=args.theme)


if __name__ == "__main__":
    main()
