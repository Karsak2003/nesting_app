#!/usr/bin/env python3
"""
ИАГИ - Интеллектуальный агент гравитационной имитации для раскроя плоских деталей
===================================================================================

Единая точка входа для запуска приложения в различных режимах:
- Консольный режим для автоматической обработки
- Графический интерфейс (legacy GUI) для визуализации и ручного управления
- Клиент-серверная архитектура (FastAPI + PyQt5)
- Пакетная обработка наборов данных
- Интеграцию с промышленными CAD/CAM-системами

Разработано в соответствии с требованиями глав 2-3 диссертации:
- Геометрическая корректность представления фигур
- Физически мотивированная динамика размещения
- Учет технологических ограничений
- Гарантированная сходимость и устойчивость
"""

import os
import sys
import time
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Настройка путей для импорта модулей
sys.path.insert(0, str(Path(__file__).parent / 'server'))

# Основные компоненты системы (из server/)
from core.geometry import PolygonShape
from core.optimizer import PackingOptimizer
from core.constraints import ConstraintManager
from config.settings import get_config, load_profile
from my_io.dxf_import import import_dxf
from my_io.step_import import import_step
from my_io.exporter import export_results
from algorithms.hybrid import hybrid_optimization_factory
from utils.polygonization import AdaptivePolygonizer

# Графический интерфейс (опционально) - legacy GUI
try:
    from gui.interface import main as gui_main
    GUI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Графический интерфейс (legacy) недоступен: {e}")
    GUI_AVAILABLE = False

# Клиент-серверные компоненты (опционально)
try:
    from client.main import run_server, run_client, run_both
    CLIENT_SERVER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Клиент-серверные компоненты недоступны: {e}")
    CLIENT_SERVER_AVAILABLE = False

# Настройка логирования
def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/app.log'):
    """Настройка системы логирования"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('IAGI-Main')

logger = setup_logging()

def load_shapes_from_file(input_file: str, tolerance: float = 0.1) -> List[PolygonShape]:
    """
    Загрузка фигур из файла в зависимости от формата
    
    :param input_file: Путь к файлу
    :param tolerance: Точность полигонализации
    :return: Список загруженных фигур
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")
    
    logger.info(f"Загрузка фигур из файла: {input_file}")
    
    try:
        # Определение формата по расширению
        if input_path.suffix.lower() in ['.dxf']:
            shapes = import_dxf(str(input_path), tolerance=tolerance)
        elif input_path.suffix.lower() in ['.step', '.stp']:
            # Получение размеров листа из конфигурации
            config = get_config()
            sheet_size = config.get_sheet_size_from_input(str(input_path))
            shapes = import_step(str(input_path), tolerance=tolerance, sheet_size=sheet_size)
        elif input_path.suffix.lower() in ['.json']:
            # Загрузка из JSON (например, для тестовых наборов)
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Здесь должна быть логика преобразования JSON в фигуры
            raise NotImplementedError("Загрузка из JSON пока не поддерживается")
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {input_path.suffix}")
        
        logger.info(f"Загружено {len(shapes)} фигур из {input_file}")
        return shapes
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке фигур из {input_file}: {e}")
        raise

def load_technological_constraints(constraint_file: Optional[str] = None,
                                  input_file: Optional[str] = None,
                                  sheet_size: Tuple[float, float] = (2000.0, 1000.0)) -> ConstraintManager:
    """
    Загрузка технологических ограничений
    
    :param constraint_file: Файл с ограничениями (DXF, JSON)
    :param input_file: Основной файл с фигурами (DXF) - для извлечения дефектных зон, если constraint_file не указан
    :param sheet_size: Размеры листа
    :return: Менеджер ограничений
    """
    constraint_manager = ConstraintManager(sheet_size)
    
    if constraint_file:
        constraint_path = Path(constraint_file)
        
        if not constraint_path.exists():
            logger.warning(f"Файл с ограничениями не найден: {constraint_file}. Используются значения по умолчанию.")
            return constraint_manager
        
        try:
            if constraint_path.suffix.lower() == '.json':
                constraint_manager.import_from_json(str(constraint_path))
            elif constraint_path.suffix.lower() == '.dxf':
                constraint_manager.import_from_dxf(str(constraint_path))
            else:
                logger.warning(f"Неподдерживаемый формат для ограничений: {constraint_path.suffix}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке ограничений из {constraint_file}: {e}")
    elif input_file and Path(input_file).suffix.lower() == '.dxf':
        try:
            constraint_manager.import_from_dxf(str(input_file))
            logger.info(f"Импортированы дефектные зоны из основного DXF файла: {input_file}")
        except Exception as e:
            logger.warning(f"Не удалось импортировать дефектные зоны из {input_file}: {e}")
    
    return constraint_manager

def run_optimization(input_file: str, 
                    output_dir: str = 'results',
                    profile: str = 'medium_precision',
                    algorithm: str = 'sequential',
                    constraint_file: Optional[str] = None,
                    visualize: bool = False,
                    use_original_positions: bool = True) -> Dict[str, Any]:
    """
    Основной метод оптимизации раскроя
    
    :param input_file: Файл с данными о фигурах (DXF/STEP)
    :param output_dir: Директория для сохранения результатов
    :param profile: Профиль конфигурации ('high_precision', 'medium_precision', 'low_precision')
    :param algorithm: Алгоритм размещения ('sequential', 'parallel', 'hybrid')
    :param constraint_file: Файл с технологическими ограничениями
    :param visualize: Визуализировать результат
    :return: Словарь с результатами оптимизации
    """
    start_time = time.time()
    logger.info(f"=== ЗАПУСК ОПТИМИЗАЦИИ ИАГИ ===")
    logger.info(f"Входной файл: {input_file}")
    logger.info(f"Алгоритм: {algorithm}, Профиль: {profile}")
    
    try:
        # Получение конфигурации
        config = get_config()
        config.apply_profile(profile)
        
        # Загрузка размеров листа из файла или использование значений по умолчанию
        sheet_size = config.get_sheet_size_from_input(input_file)
        logger.info(f"Размеры листа: {sheet_size[0]}x{sheet_size[1]} мм")
        
        # Загрузка технологических ограничений
        constraint_manager = load_technological_constraints(constraint_file, input_file, sheet_size)
        
        # Установка технологии резки из конфигурации
        technology = config.technological_constraints['cutting_technology']
        constraint_manager.set_technology(technology)
        min_gap = constraint_manager.current_gap
        
        # Загрузка фигур с адаптивной полигонализацией
        tolerance = config.geometry['polygonization']['tolerance']
        shapes = load_shapes_from_file(input_file, tolerance=tolerance)
        
        # Проверка наличия фигур
        if not shapes:
            raise ValueError("Не загружено ни одной фигуры для оптимизации")

        for i, shape in enumerate(shapes):
            shape_name = shape.name
            if hasattr(shape, 'priority') and shape.priority is not None:
                constraint_manager.set_placement_priority(shape_name, shape.priority)
            if hasattr(shape, 'allowed_angles') and shape.allowed_angles is not None:
                constraint_manager.add_orientation_constraint(shape_name, shape.allowed_angles)
        
        # Создание оптимизатора
        optimizer = PackingOptimizer({
            'sheet_size': sheet_size,
            'min_gap': min_gap,
            'time_limit': config.system['max_execution_time'],
            'use_sequential': algorithm == 'sequential',
            'stabilization_enabled': True
        })
        
        # Добавление дефектных зон из менеджера ограничений
        for defect in constraint_manager.defect_zones:
            contour = list(defect['polygon'].exterior.coords)[:-1]
            optimizer.add_defect_zone(contour)

        logger.info(f"Режим начальных позиций: {'исходные позиции из DXF' if use_original_positions else 'автоматическое размещение'}")
        result_agents = optimizer.optimize(
            shapes=shapes,
            priorities=[constraint_manager.get_shape_priority(shape.name) 
                       for shape in shapes],
            orientation_constraints=[constraint_manager.orientation_constraints.get(shape.name, None)
                                    for shape in shapes],
            use_original_positions=use_original_positions
        )
        
        # Подготовка данных для экспорта
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
        
        # Экспорт результатов

        base_output_dir = str(Path(output_dir).resolve())
        Path(base_output_dir).mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Сохранение результатов в директорию: {base_output_dir}")
        
        exported_files = export_results(
            placements=placements,
            output_dir=base_output_dir,
            sheet_size=sheet_size,
            min_gap=min_gap,
            defect_zones=[PolygonShape(zone['polygon'].exterior.coords, name=f"defect_{i}") 
                         for i, zone in enumerate(constraint_manager.defect_zones)],
            constraint_manager=constraint_manager,
            formats=['dxf', 'svg', 'json'],
            create_timestamped_folder=True
        )
        
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
        if max_x > min_x and max_y > min_y:
            effective_area = (max_x - min_x) * (max_y - min_y)
        else:
            effective_area = sheet_size[0] * max_y if max_y > 0 else 1.0
        utilization = (total_area / effective_area) * 100 if effective_area > 0 else 0.0
        # Формирование отчета
        result_report = {
            'input_file': input_file,
            'output_dir': output_dir,
            'algorithm': algorithm,
            'profile': profile,
            'sheet_size': sheet_size,
            'num_shapes': len(shapes),
            'utilization': utilization,
            'execution_time': time.time() - start_time,
            'exported_files': exported_files,
            'constraints': {
                'technology': technology,
                'min_gap': min_gap,
                'num_defect_zones': len(constraint_manager.defect_zones)
            }
        }
        
        logger.info(f"Оптимизация завершена за {result_report['execution_time']:.2f} секунд")
        logger.info(f"Коэффициент использования материала: {utilization:.2f}%")
        
        actual_output_dir = None
        if exported_files:

            first_file = list(exported_files.values())[0]
            actual_output_dir = os.path.dirname(first_file)
        
        if actual_output_dir:
            logger.info(f"Результаты сохранены в: {os.path.abspath(actual_output_dir)}")
            print(f"\n{'='*60}")
            print(f"РЕЗУЛЬТАТЫ УСПЕШНО СОХРАНЕНЫ")
            print(f"{'='*60}")
            print(f"Директория: {os.path.abspath(actual_output_dir)}")
            print(f"Экспортированные файлы:")
            for fmt, path in exported_files.items():
                print(f"  - {fmt.upper()}: {os.path.basename(path)}")
            print(f"{'='*60}\n")
        else:
            logger.warning(f"Не удалось определить директорию сохранения результатов")
        
        # Визуализация (если запущен GUI)
        if visualize and GUI_AVAILABLE:
            try:
                # Передача данных в GUI для визуализации
                gui_data = {
                    'placements': placements,
                    'sheet_size': sheet_size,
                    'defect_zones': constraint_manager.defect_zones,
                    'constraint_manager': constraint_manager
                }
                # Здесь можно запустить GUI с передачей данных
                # gui_main(gui_data)
                pass
            except Exception as e:
                logger.warning(f"Не удалось запустить визуализацию: {e}")
        
        return result_report
        
    except Exception as e:
        logger.error(f"Критическая ошибка при оптимизации: {e}")
        raise

def run_batch_optimization(config_file: str):
    """
    Пакетная обработка набора задач раскроя
    
    :param config_file: JSON-файл с конфигурацией пакетной обработки
    """
    logger.info(f"Запуск пакетной обработки из файла: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            batch_config = json.load(f)
        
        results = []
        total_start_time = time.time()
        
        for task in batch_config.get('tasks', []):
            task_start_time = time.time()
            logger.info(f"Обработка задачи: {task.get('name', 'unnamed')}")
            
            try:
                result = run_optimization(
                    input_file=task['input_file'],
                    output_dir=task.get('output_dir', f"results/{task.get('name', 'task')}"),
                    profile=task.get('profile', 'medium_precision'),
                    algorithm=task.get('algorithm', 'sequential'),
                    constraint_file=task.get('constraint_file'),
                    visualize=False
                )
                results.append(result)
                
                task_time = time.time() - task_start_time
                logger.info(f"Задача '{task.get('name', 'unnamed')}' выполнена за {task_time:.2f} секунд")
            
            except Exception as e:
                logger.error(f"Ошибка при обработке задачи '{task.get('name', 'unnamed')}': {e}")
                results.append({
                    'task_name': task.get('name', 'unnamed'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Генерация сводного отчета
        total_time = time.time() - total_start_time
        report = {
            'batch_name': batch_config.get('name', 'unnamed_batch'),
            'total_tasks': len(batch_config.get('tasks', [])),
            'successful_tasks': sum(1 for r in results if r.get('status') != 'failed'),
            'total_time': total_time,
            'results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Сохранение отчета
        report_file = Path(batch_config.get('report_dir', 'reports')) / f"batch_report_{int(time.time())}.json"
        report_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Пакетная обработка завершена. Сводный отчет сохранен в {report_file}")
        return report
    
    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке: {e}")
        raise

def create_test_data(num_shapes: int = 10) -> List[PolygonShape]:
    """
    Генерация тестовых данных для демонстрации
    
    :param num_shapes: Число тестовых фигур
    :return: Список тестовых фигур
    """
    logger.info(f"Генерация тестовых данных ({num_shapes} фигур)...")
    
    shapes = []
    
    # Прямоугольные детали
    for i in range(num_shapes // 2):
        width = np.random.uniform(50, 150)
        height = np.random.uniform(30, 100)
        contour = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]
        shape = PolygonShape(contour, name=f"rect_{i}")
        shape.priority = 1 if i < num_shapes // 4 else 2
        if i < num_shapes // 4:
            shape.allowed_angles = [0, 90]  # Ограничение на ориентацию
        shapes.append(shape)
    
    # L-образные детали
    for i in range(num_shapes // 2, num_shapes - 1):
        outer = [
            (0, 0), (100, 0), (100, 30), 
            (70, 30), (70, 100), (0, 100)
        ]
        shape = PolygonShape(outer, name=f"L_shape_{i}")
        shape.priority = 1
        shapes.append(shape)
    
    # Деталь с отверстием
    outer = [(0, 0), (120, 0), (120, 80), (0, 80)]
    hole = [(40, 20), (80, 20), (80, 60), (40, 60)]
    shape = PolygonShape(outer, [hole], name="part_with_hole")
    shape.priority = 1
    shapes.append(shape)
    
    logger.info(f"Сгенерировано {len(shapes)} тестовых фигур")
    return shapes

def main():
    """Основная точка входа приложения"""
    parser = argparse.ArgumentParser(
        description='ИАГИ - Система раскроя плоских деталей на основе интеллектуальных агентов гравитационной имитации',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Консольный режим (локальная оптимизация)
  python main.py --input parts.dxf --output results
  
  # Legacy GUI (старый графический интерфейс)
  python main.py --gui
  
  # Клиент-серверный режим (новая архитектура)
  python main.py --server              # Только сервер FastAPI
  python main.py --client              # Только клиент PyQt5
  python main.py --both                # Сервер и клиент вместе
  
  # Расширенные настройки сервера
  python main.py --server --host 0.0.0.0 --port 8080
  
  # Расширенные настройки клиента
  python main.py --client --url http://192.168.1.100:8000 --theme dark
  
  # Пакетная обработка
  python main.py --batch config.json
  
  # Тестовый запуск
  python main.py --test --num_shapes 20
        """
    )
    
    # Режимы работы - основные
    mode_group = parser.add_argument_group('Режимы работы')
    mode_group.add_argument('--gui', action='store_true', help='Запустить legacy графический интерфейс (gui/interface.py)')
    mode_group.add_argument('--batch', type=str, metavar='CONFIG_FILE', help='Запустить пакетную обработку из JSON конфигурации')
    mode_group.add_argument('--test', action='store_true', help='Запустить на тестовых данных')
    
    # Режимы работы - клиент-сервер
    cs_mode_group = parser.add_argument_group('Клиент-серверные режимы (новая архитектура)')
    cs_mode_group.add_argument('--server', action='store_true',
                               help='Запустить только серверную часть (FastAPI)')
    cs_mode_group.add_argument('--client', action='store_true',
                               help='Запустить только клиентскую часть (PyQt5)')
    cs_mode_group.add_argument('--both', action='store_true',
                               help='Запустить обе части одновременно')
    
    # Параметры для консольного режима
    console_group = parser.add_argument_group('Параметры консольного режима')
    console_group.add_argument('--input', type=str, metavar='FILE', help='Входной файл с данными (DXF/STEP)')
    console_group.add_argument('--output', type=str, default='results', metavar='DIR', help='Директория для результатов')
    console_group.add_argument('--constraints', type=str, metavar='FILE', help='Файл с технологическими ограничениями')
    console_group.add_argument('--profile', type=str, default='medium_precision', 
                              choices=['high_precision', 'medium_precision', 'low_precision'],
                              help='Профиль конфигурации')
    console_group.add_argument('--algorithm', type=str, default='sequential',
                              choices=['sequential', 'parallel', 'hybrid'],
                              help='Алгоритм размещения')
    console_group.add_argument('--visualize', action='store_true', help='Визуализировать результат')
    console_group.add_argument('--optimize-positions', action='store_true', 
                              help='Оптимизировать размещение фигур (по умолчанию используются исходные позиции из DXF)')
    
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
    
    # Тестовые параметры
    test_group = parser.add_argument_group('Параметры тестового режима')
    test_group.add_argument('--num_shapes', type=int, default=10, help='Число тестовых фигур')
    
    # Логирование
    log_group = parser.add_argument_group('Параметры логирования')
    log_group.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Уровень детализации логирования')
    log_group.add_argument('--log_file', type=str, default='logs/app.log', help='Файл лога')
    
    args = parser.parse_args()
    
    # Настройка логирования
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        logger.info("=== СИСТЕМА РАСКРОЯ ПЛОСКИХ ДЕТАЛЕЙ НА ОСНОВЕ ИАГИ ===")
        logger.info(f"Версия: 1.0")
        logger.info(f"Текущее время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Проверка доступности клиент-серверных компонентов
        if (args.server or args.client or args.both) and not CLIENT_SERVER_AVAILABLE:
            logger.error("Клиент-серверные компоненты недоступны. Установите зависимости: fastapi, uvicorn, PyQt5, matplotlib")
            return 1
        
        # Клиент-серверные режимы (приоритет)
        if args.both:
            logger.info("Запуск сервера и клиента одновременно...")
            run_both(host=args.host, port=args.port, theme=args.theme)
            return 0
        elif args.server:
            logger.info(f"Запуск сервера на {args.host}:{args.port}...")
            run_server(host=args.host, port=args.port, reload=args.reload)
            return 0
        elif args.client:
            logger.info("Запуск клиента...")
            run_client(server_url=args.url, theme=args.theme)
            return 0
        
        # Режим графического интерфейса (legacy)
        if args.gui:
            if not GUI_AVAILABLE:
                logger.error("Графический интерфейс (legacy) недоступен. Проверьте установку зависимостей PyQt5.")
                return 1
            
            logger.info("Запуск legacy графического интерфейса...")
            return gui_main()
        
        # Режим пакетной обработки
        if args.batch:
            return run_batch_optimization(args.batch)
        
        # Тестовый режим
        if args.test:
            # Генерация тестовых данных
            shapes = create_test_data(args.num_shapes)
            
            # Создание временного файла
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                test_data = {
                    'shapes': [{
                        'contour': shape.outer_contour.tolist(),
                        'inner_contours': [contour.tolist() for contour in shape.inner_contours],
                        'name': shape.name,
                        'priority': getattr(shape, 'priority', 1)
                    } for shape in shapes]
                }
                json.dump(test_data, f)
                temp_file = f.name
            
            try:
                # Запуск оптимизации на тестовых данных
                result = run_optimization(
                    input_file=temp_file,
                    output_dir='results/test',
                    profile=args.profile,
                    algorithm=args.algorithm,
                    visualize=args.visualize
                )
                return 0
            finally:
                # Очистка временного файла
                os.unlink(temp_file)
        
        # Консольный режим с указанием входного файла
        if args.input:
            if not Path(args.input).exists():
                logger.error(f"Входной файл не найден: {args.input}")
                return 1
            
            use_original_positions = not args.optimize_positions
            logger.info(f"Параметр --optimize-positions: {args.optimize_positions}")
            logger.info(f"Режим use_original_positions: {use_original_positions}")
            
            result = run_optimization(
                input_file=args.input,
                output_dir=args.output,
                profile=args.profile,
                algorithm=args.algorithm,
                constraint_file=args.constraints,
                visualize=args.visualize,
                use_original_positions=use_original_positions
            )
            return 0
        
        # Если не указан режим работы - показать справку
        parser.print_help()
        return 1
        
    except KeyboardInterrupt:
        logger.info("Выполнение прервано пользователем")
        return 1
    except Exception as e:
        logger.exception(f"Необработанное исключение: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())