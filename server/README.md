# Серверная часть ИАГИ

Серверное приложение на FastAPI для оптимизации раскроя плоских деталей.

## Установка зависимостей

```bash
pip install fastapi uvicorn python-multipart requests numpy scipy shapely rtree matplotlib ezdxf pyyaml scikit-learn tqdm joblib
```

## Запуск сервера

```bash
cd server
python main.py
```

Или через uvicorn напрямую:

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /` - Информация о сервисе
- `GET /health` - Проверка здоровья сервиса
- `POST /api/optimize` - Запуск оптимизации
- `GET /api/status/{task_id}` - Получение статуса задачи
- `GET /api/download/{task_id}/{format}` - Скачивание результата
- `POST /api/shapes/info` - Получение информации о фигурах
- `GET /api/profiles` - Доступные профили конфигурации
- `GET /api/algorithms` - Доступные алгоритмы

## Документация API

После запуска сервера документация Swagger доступна по адресу:
http://localhost:8000/docs

ReDoc документация:
http://localhost:8000/redoc
