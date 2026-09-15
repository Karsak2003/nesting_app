# ИАГИ - Разделение на серверную и клиентскую части

Проект разделен на две независимые части: серверную (FastAPI) и клиентскую (PySide6 + HTTP).

## Структура проекта

```
/workspace/
├── server/              # Серверная часть
│   ├── config/          # Конфигурация
│   ├── my_io/           # Модули ввода-вывода
│   ├── utils/           # Утилиты
│   ├── algorithms/      # Алгоритмы оптимизации
│   ├── core/            # Общие модули ядра
│   ├── main.py          # FastAPI приложение
│   └── README.md        # Документация сервера
│
├── client/              # Клиентская часть  
│   ├── gui/             # Оригинальный GUI (для локального запуска)
│   ├── config/          # Конфигурация
│   ├── main.py          # PyQt5 приложение
│   └── README.md        # Документация клиента
│
└── requirements.txt     # Зависимости
```

## Запуск серверной части

Сервер предоставляет REST API для выполнения оптимизации раскроя.

```bash
cd /workspace/server
python main.py
```

Или через uvicorn:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

**API документация:** http://localhost:8000/docs

## Запуск клиентской части

Клиентское приложение с графическим интерфейсом для работы с сервером.

```bash
cd /workspace/client
python main.py
```

### Настройка подключения

В клиентском приложении можно настроить адрес сервера:
1. Нажмите кнопку "🔌 Подключение..." в верхней панели
2. Введите адрес сервера (по умолчанию `http://localhost:8000`)
3. Выберите пресет "Локальный" или "Удаленный"
4. Нажмите "OK"

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Необходимые пакеты:
- **Для сервера:** fastapi, uvicorn, python-multipart
- **Для клиента:** PyQt5, requests
- **Общие:** numpy, scipy, shapely, rtree, matplotlib, ezdxf, pyyaml

## Архитектура взаимодействия

```
┌─────────────────┐      HTTP/REST API      ┌──────────────────┐
│ Клиент (PySide6)│ ◄────────────────────►  │  Сервер (FastAPI)│
│                 │                         │                  │
│ - Выбор файла   │    1. Загрузка файла    │ - Прием файлов   │
│ - Настройки     │    2. Запуск задачи     │ - Оптимизация    │
│ - Мониторинг    │    3. Поллинг статуса   │ - Хранение задач │
│ - Экспорт       │    4. Скачивание        │ - Экспорт        │
└─────────────────┘                         └──────────────────┘
```

## Сценарии использования

### 1. Локальная работа (сервер + клиент на одной машине)

```bash
# Терминал 1: запуск сервера
cd /workspace/server
python main.py

# Терминал 2: запуск клиента
cd /workspace/client
python main.py
```

### 2. Удаленная работа (клиент подключается к удаленному серверу)

```bash
# На сервере:
cd /workspace/server
python main.py  # или uvicorn server.main:app --host 0.0.0.0 --port 8000

# На клиенте:
cd /workspace/client
python main.py
# В интерфейсе указать адрес сервера: http://<server-ip>:8000
```

### 3. Только сервер (для интеграции с другими системами)

```bash
cd /workspace/server
python main.py
# Используйте REST API напрямую через curl, Postman или другую систему
```

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Информация о сервисе |
| GET | `/health` | Проверка здоровья |
| POST | `/api/optimize` | Запуск оптимизации |
| GET | `/api/status/{task_id}` | Статус задачи |
| GET | `/api/download/{task_id}/{format}` | Скачивание результата |
| POST | `/api/shapes/info` | Информация о фигурах |
| GET | `/api/profiles` | Доступные профили |
| GET | `/api/algorithms` | Доступные алгоритмы |

## Пример использования API

```bash
# Запуск оптимизации
curl -X POST "http://localhost:8000/api/optimize" \
  -F "file=@parts.dxf" \
  -F "profile=medium_precision" \
  -F "algorithm=sequential" \
  -F "min_gap=0.5" \
  -F "sheet_width=2000" \
  -F "sheet_height=1000"

# Получение статуса
curl "http://localhost:8000/api/status/{task_id}"

# Скачивание результата
curl "http://localhost:8000/api/download/{task_id}/dxf" -o result.dxf
```
