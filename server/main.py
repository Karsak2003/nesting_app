"""
Серверная часть приложения ИАГИ на FastAPI
Предоставляет API для оптимизации раскроя плоских деталей
"""
import sys, asyncio

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        print("[IAGI] Ok: Установлен WindowsSelectorEventLoopPolicy", flush=True)
    except Exception as e:
        print(f"[IAGI] WARNING: Не удалось установить SelectorEventLoop: {e}", flush=True)

from pydantic import BaseModel, Field
from pathlib import Path
import logging

# Добавляем корень проекта в путь импорта
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / 'server.log')
    ]
)
logger = logging.getLogger('IAGI-Server')

from app import (
    # --- Стандартные библиотеки ---
    os, time, json, threading, tempfile, shutil, uuid,
    List, Dict, Tuple, Optional, Any, asynccontextmanager,
    
    # --- Сторонние библиотеки ---
    np, FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Query,
    JSONResponse, FileResponse, uvicorn,
    
    # --- Локальные модули (Core, IO, Config) ---
    PolygonShape, PackingOptimizer, ConstraintManager, ProgressTracker,
    import_dxf, import_step, export_results, get_config, load_profile
)


# Хранилище задач оптимизации
optimization_tasks: Dict[str, Dict[str, Any]] = {}

# Временная директория для файлов
TEMP_DIR = Path(tempfile.gettempdir()) / 'iagi_server'
TEMP_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Диагностика event loop
    loop = asyncio.get_running_loop()
    loop_type = type(loop).__name__
    logger.info(f"Запуск сервера ИАГИ...")
    logger.info(f"[ДИАГНОСТИКА] Event loop: {loop_type}")
    logger.info(f"[ДИАГНОСТИКА] Python: {sys.version}")
    logger.info(f"[ДИАГНОСТИКА] Platform: {sys.platform}")
    
    if sys.platform == "win32" and "Selector" not in loop_type:
        logger.warning(
            f"[ДИАГНОСТИКА] WARNING: Используется {loop_type}, а не SelectorEventLoop! "
            f"Это может вызывать WinError 10014."
        )
    
    yield
    logger.info("Остановка сервера ИАГИ...")


app = FastAPI(
    title=      "ИАГИ API",
    description="API для интеллектуального агента гравитационной имитации раскроя",
    version=    "1.0.0",
    lifespan=   lifespan
)


# === Модели данных ===

class OptimizationRequest(BaseModel):
    """Запрос на оптимизацию"""
    profile:                    str =   Field(default='medium_precision', description='Профиль конфигурации')
    algorithm:                  str =   Field(default='sequential', description='Алгоритм размещения')
    technology:                 str =   Field(default='laser', description='Технология резки')
    min_gap:                    float = Field(default=0.5, description='Минимальный зазор (мм)')
    time_limit:                 int =   Field(default=300, description='Лимит времени (секунды)')
    sheet_width:                float = Field(default=2000.0, description='Ширина листа (мм)')
    sheet_height:               float = Field(default=1000.0, description='Высота листа (мм)')
    use_original_positions:     bool =  Field(default=True, description='Использовать исходные позиции')
    history_sample_rate:        int =   Field( default=5, description='Сохранять снимок current_shapes каждые N итераций')
    progress_callback_interval: int =   Field( default=20, description='Частота записи в history (в итерациях симуляции)' )


class OptimizationResponse(BaseModel):
    """Ответ оптимизации"""
    task_id: str
    status: str
    message: str
    utilization: Optional[float] = None
    execution_time: Optional[float] = None
    result_files: Optional[Dict[str, str]] = None


class TaskStatus(BaseModel):
    """Статус задачи"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    message: str
    utilization: Optional[float] = None
    error: Optional[str] = None
    history: Optional[Dict[str, List]] = None
    current_shapes: Optional[List[Dict[str, Any]]] = None  # <-- НОВОЕ: снимок фигур


class ShapeInfo(BaseModel):
    """Информация о фигуре"""
    name: str
    area: float
    priority: Optional[int] = None
    bounding_box: Tuple[float, float, float, float]


# === Вспомогательные функции ===

def load_shapes_from_file(input_file: str, tolerance: float = 0.1) -> List[PolygonShape]:
    """Загрузка фигур из файла"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")
    
    logger.info(f"Загрузка фигур из файла: {input_file}")
    
    if input_path.suffix.lower() in ['.dxf']:
        shapes = import_dxf(str(input_path), tolerance=tolerance)
    elif input_path.suffix.lower() in ['.step', '.stp']:
        config = get_config()
        sheet_size = (2000.0, 1000.0)  # Значение по умолчанию
        shapes = import_step(str(input_path), tolerance=tolerance, sheet_size=sheet_size)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {input_path.suffix}")
    
    logger.info(f"Загружено {len(shapes)} фигур")
    return shapes


def run_optimization_task(
    task_id: str,
    shapes: List[PolygonShape],
    request: OptimizationRequest,
    constraint_manager: ConstraintManager
):
    
    """Выполнение задачи оптимизации в фоновом режиме"""
    try:
        optimization_tasks[task_id]['status'] = 'running'
        optimization_tasks[task_id]['progress'] = 0
        
        start_time = time.time()
        task_lock = threading.Lock()
        
        tracker = ProgressTracker(
            task_id=task_id,
            task_dict=optimization_tasks[task_id],      # ✅ Передаём весь словарь задачи
            lock=task_lock,
            snapshot_interval=request.history_sample_rate,       # ✅ Частота снимков
            callback_interval=request.progress_callback_interval # ✅ Частота записи в history
        )
        
        # Создание оптимизатора
        optimizer = PackingOptimizer({
            'sheet_size': (request.sheet_width, request.sheet_height),
            'min_gap': request.min_gap,
            'time_limit': request.time_limit,
            'use_sequential': request.algorithm == 'sequential',
            'stabilization_enabled': True
        })
        
        # Добавление дефектных зон
        for defect in constraint_manager.defect_zones:
            contour = list(defect['polygon'].exterior.coords)[:-1]
            optimizer.add_defect_zone(contour)
        
        # Запуск оптимизации с отслеживанием прогресса
        result_agents = optimizer.optimize(
            shapes=shapes,
            priorities=[constraint_manager.get_shape_priority(shape.name) for shape in shapes],
            orientation_constraints=[
                constraint_manager.orientation_constraints.get(shape.name, None)
                for shape in shapes
            ],
            use_original_positions=request.use_original_positions,
            progress_callback=tracker
        )
        
        
        # Подготовка результатов
        placements = []
        for agent in result_agents:
            transformed_shape = agent.get_transformed_shape()
            placements.append({
                'shape': transformed_shape,
                'position': agent.position,
                'angle': agent.angle,
                'name': getattr(agent.shape, 'name', f'part_{len(placements)+1}'),
                'priority': agent.priority
            })
        
        # Расчет метрик
        total_area = sum(agent.shape.area for agent in result_agents)
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for agent in result_agents:
            transformed_shape = agent.get_transformed_shape()
            x_min, y_min, x_max, y_max = transformed_shape.get_bounding_box()
            min_x = min(min_x, x_min)
            min_y = min(min_y, y_min)
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)
        
        # Утилизация считается относительно площади листа, а не bounding box фигур
        sheet_area = request.sheet_width * request.sheet_height
        utilization = (total_area / sheet_area) * 100 if sheet_area > 0 else 0.0
        
        execution_time = time.time() - start_time
        
        # Экспорт результатов во временную директорию
        output_dir = TEMP_DIR / task_id
        output_dir.mkdir(exist_ok=True)
        
        exported_files = export_results(
            placements=placements,
            output_dir=str(output_dir),
            sheet_size=(request.sheet_width, request.sheet_height),
            min_gap=request.min_gap,
            defect_zones=[
                PolygonShape(zone['polygon'].exterior.coords, name=f"defect_{i}")
                for i, zone in enumerate(constraint_manager.defect_zones)
            ],
            constraint_manager=constraint_manager,
            formats=['dxf', 'svg', 'json'],
            create_timestamped_folder=False
        )
        
        # Обновление статуса задачи
        optimization_tasks[task_id].update({
            'status': 'completed',
            'progress': 100,
            'utilization': utilization,
            'execution_time': execution_time,
            'result_files': {k: str(v) for k, v in exported_files.items()},
            'num_shapes': len(shapes),
            'message': f'Оптимизация завершена. Использование: {utilization:.2f}%'
        })
        
        logger.info(f"Задача {task_id} завершена. Использование: {utilization:.2f}%")
        
    except Exception as e:
        logger.error(f"Ошибка в задаче {task_id}: {e}")
        optimization_tasks[task_id].update({
            'status': 'failed',
            'error': str(e),
            'message': f'Ошибка: {str(e)}'
        })


def update_task_progress(task_id: str, progress: int, utilization: float, energy: float = 0.0):
    """Обновление прогресса задачи и истории"""
    if task_id in optimization_tasks:
        optimization_tasks[task_id]['progress'] = min(100, progress)
        optimization_tasks[task_id]['utilization'] = utilization
        
        # Обновление истории (с прореживанием, чтобы не перегружать память и сеть)
        hist = optimization_tasks[task_id]['history']
        iteration_num = len(hist['iterations']) + 1
        
        # Сохраняем каждую 5-ю итерацию или если это последняя точка (progress == 100)
        if iteration_num % 5 == 0 or progress >= 100:
            hist['iterations'].append(iteration_num)
            hist['energies'].append(round(energy, 4))
            hist['utilizations'].append(round(utilization, 4))

# === API Endpoints ===

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "ИАГИ API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "optimize": "/api/optimize",
            "status": "/api/status/{task_id}",
            "download": "/api/download/{task_id}/{format}",
            "shapes": "/api/shapes/info"
        }
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy"}


@app.post("/api/optimize", response_model=OptimizationResponse)
async def start_optimization(
    file:                       UploadFile = File(..., description="Файл с фигурами (DXF, STEP)"),
    profile:                    str =        Form(default='medium_precision'),
    algorithm:                  str =        Form(default='sequential'),
    technology:                 str =        Form(default='laser'),
    min_gap:                    float =      Form(default=0.5),
    time_limit:                 int =        Form(default=300),
    sheet_width:                float =      Form(default=2000.0),
    sheet_height:               float =      Form(default=1000.0),
    use_original_positions:     bool =       Form(default=True),
    history_sample_rate:        int =        Form(default=5),
    progress_callback_interval: int =        Form(default=20)
):
    """
    Запуск оптимизации раскроя
    
    - **file**: Файл с фигурами в формате DXF или STEP
    - **profile**: Профиль конфигурации (high_precision, medium_precision, low_precision)
    - **algorithm**: Алгоритм размещения (sequential, parallel, hybrid)
    - **technology**: Технология резки (laser, plasma, waterjet)
    - **min_gap**: Минимальный зазор между деталями в мм
    - **time_limit**: Максимальное время оптимизации в секундах
    - **sheet_width**: Ширина листа в мм
    - **sheet_height**: Высота листа в мм
    - **use_original_positions**: Использовать ли исходные позиции из файла
    """
    import uuid
    
    task_id = str(uuid.uuid4())
    
    # Сохранение загруженного файла
    file_path = TEMP_DIR / f"{task_id}_{file.filename}"
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Ошибка сохранения файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сохранения файла")
    
    # Создание запроса
    request = OptimizationRequest(
        profile=profile,
        algorithm=algorithm,
        technology=technology,
        min_gap=min_gap,
        time_limit=time_limit,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        use_original_positions=use_original_positions,
        history_sample_rate=history_sample_rate,          
        progress_callback_interval=progress_callback_interval  
    )
    
    # Инициализация задачи
    optimization_tasks[task_id] = {
        'task_id': task_id,
        'status': 'pending',
        'progress': 0,
        'message': 'Загрузка и обработка файла...',
        'file_path': str(file_path),
        'history': {                     # <--- ДОБАВЛЕНО
            'iterations': [],
            'energies': [],
            'utilizations': []
        },
        'current_shapes': None
    }
    
    try:
        # Загрузка фигур
        tolerance = 0.1 if profile == 'high_precision' else (0.5 if profile == 'low_precision' else 0.2)
        shapes = load_shapes_from_file(str(file_path), tolerance=tolerance)
        
        if not shapes:
            raise ValueError("Не загружено ни одной фигуры")
        
        # Создание менеджера ограничений
        constraint_manager = ConstraintManager((request.sheet_width, request.sheet_height))
        constraint_manager.set_technology(request.technology)
        
        # Запуск фонового задания
        async def _run_background():
            """Обёртка для запуска синхронной задачи в отдельном потоке"""
            try:
                await asyncio.to_thread(
                    run_optimization_task,
                    task_id, shapes, request, constraint_manager
                )
            except Exception as e:
                logger.error(f"Ошибка в фоновой задаче {task_id}: {e}")
                
        asyncio.create_task(_run_background())
        
        return OptimizationResponse(
            task_id=task_id,
            status='pending',
            message='Задача оптимизации запущена'
        )
        
    except Exception as e:
        logger.error(f"Ошибка запуска оптимизации: {e}")
        optimization_tasks[task_id]['status'] = 'failed'
        optimization_tasks[task_id]['error'] = str(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str,
    since_iteration: Optional[int] = Query(
        default=None,
        description="Вернуть историю только после этой итерации (дельта). "
                    "Если не указан — возвращается полная история."
    )
):
    """
    Получение статуса задачи оптимизации.
    Если указан since_iteration, возвращается только дельта истории 
    (итерации > since_iteration), что уменьшает payload.
    """
    if task_id not in optimization_tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = optimization_tasks[task_id]
    history = task.get('history', {})
    
    # Если запрошена дельта — фильтруем историю
    if (since_iteration is not None) and history:
        iterations = history.get('iterations', [])
        energies = history.get('energies', [])
        utilizations = history.get('utilizations', [])
        
        # Находим индекс, с которого начинаются новые данные
        start_idx = 0
        for i, it in enumerate(iterations):
            if it > since_iteration:
                start_idx = i
                break
        
        # Создаём дельту
        history_delta = {
            'iterations': iterations[start_idx:],
            'energies': energies[start_idx:],
            'utilizations': utilizations[start_idx:]
        }
    else:
        # Возвращаем полную историю (для обратной совместимости)
        history_delta = history
    
    return TaskStatus(
        task_id=task['task_id'],
        status=task['status'],
        progress=task.get('progress', 0),
        message=task.get('message', ''),
        utilization=task.get('utilization'),
        error=task.get('error'),
        history=history_delta,
        current_shapes=task.get('current_shapes')  # ← всегда возвращается
    )

@app.get("/api/download/{task_id}/{format}")
async def download_result(task_id: str, format: str):
    """Скачивание результата оптимизации"""
    if task_id not in optimization_tasks:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    task = optimization_tasks[task_id]
    
    if task['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Задача не завершена. Статус: {task['status']}"
        )
    
    result_files = task.get('result_files', {})
    if format not in result_files:
        raise HTTPException(status_code=404, detail=f"Формат {format} не найден")
    
    file_path = Path(result_files[format])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        path=str(file_path),
        filename=f"iagi_result_{task_id}.{format}",
        media_type='application/octet-stream'
    )


@app.get("/api/shapes/info")
async def get_shapes_info(file: UploadFile = File(...)):
    """Получение информации о фигурах в файле"""
    
    temp_id = str(uuid.uuid4())
    file_path = TEMP_DIR / f"{temp_id}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        shapes = load_shapes_from_file(str(file_path), tolerance=0.2)
        
        shape_infos = []
        for shape in shapes:
            bbox = shape.get_bounding_box()
            shape_infos.append({
                "name": getattr(shape, 'name', 'unknown'),
                "area": shape.area,
                "priority": getattr(shape, 'priority', None),
                "bounding_box": bbox
            })
        
        return {
            "filename": file.filename,
            "num_shapes": len(shapes),
            "shapes": shape_infos
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()


@app.post("/api/shapes/info")
async def post_shapes_info(file: UploadFile = File(...)):
    """POST версия для получения информации о фигурах (совместимость)"""
    return await get_shapes_info(file)


@app.get("/api/profiles")
async def get_profiles():
    """Получение доступных профилей конфигурации"""
    return {
        "profiles": [
            {
                "name": "high_precision",
                "description": "Авиация, космос (точность 0.05 мм)",
                "tolerance": 0.05
            },
            {
                "name": "medium_precision",
                "description": "Машиностроение (точность 0.1 мм)",
                "tolerance": 0.2
            },
            {
                "name": "low_precision",
                "description": "Текстиль, картон (точность 0.5 мм)",
                "tolerance": 0.5
            }
        ]
    }


@app.get("/api/algorithms")
async def get_algorithms():
    """Получение доступных алгоритмов"""
    return {
        "algorithms": [
            {
                "name": "sequential",
                "description": "Последовательное размещение по приоритету"
            },
            {
                "name": "parallel",
                "description": "Параллельное размещение с максимизацией плотности"
            },
            {
                "name": "hybrid",
                "description": "Гибридный алгоритм (ГА-ИАГИ, PSO-ИАГИ)"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",  # <-- Явное указание использовать стандартный asyncio loop
        # loop="uvloop",  # Альтернатива: ultra-fast loop (требует pip install uvloop)
    )
    
    server = uvicorn.Server(config)
    
    server.run()
