import numpy as np
import time
import logging
from pathlib import Path
from core.geometry import PolygonShape
from core.agent import IAGIAgent
from core.collision import CollisionDetector
from core.dynamics import GravitationalDynamics
from core.sheet_batch import Sheet, SheetBatchManager
from algorithms.stabilization import apply_stabilization
from algorithms.sequential import sequential_placement
from algorithms.parallel import parallel_placement


from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class PackingOptimizer:
    """
    Основной оптимизатор для задач раскроя-упаковки
    """
    def __init__(self, config=None):
        """
        Инициализация оптимизатора
        """
        self.config = config or {
            'sheet_size': (2000, 1000),  # мм
            'min_gap': 1.0,              # мм
            'time_limit': 600,           # секунд
            'use_sequential': True,      # Использовать последовательный алгоритм
            'stabilization_enabled': True
        }
        
        self.sheet_size = self.config['sheet_size']
        self.min_gap = self.config['min_gap']
        self.time_limit = self.config['time_limit']
        self.use_sequential = self.config['use_sequential']
        self.stabilization_enabled = self.config['stabilization_enabled']
        
        # Инициализация компонентов
        self.collision_detector = CollisionDetector(resolution=1.0)
        self.dynamics = GravitationalDynamics(
            gravity_strength=9.8,
            damping_linear=0.3,
            damping_angular=0.2
        )
        
        self.agents = []
        self.defect_zones = []
        self.result = None
        
    def add_defect_zone(self, contour):
        """Добавление дефектной зоны на лист"""
        defect_shape = PolygonShape(contour)
        self.defect_zones.append(defect_shape)
        
    def prepare_agents(self, shapes, priorities=None, orientation_constraints=None, use_original_positions=True):
        """
        Подготовка агентов для фигур
        """
        logger.info(f"=== Подготовка {len(shapes)} агентов для размещения ===")
        logger.info(f"Размер листа: {self.sheet_size[0]} x {self.sheet_size[1]} мм, min_gap={self.min_gap} мм")
        
        if use_original_positions:
            has_original_positions = any(hasattr(shape, 'original_position') for shape in shapes)
            if has_original_positions:
                logger.info("Использование исходных позиций из DXF файла")
                use_original_positions = True
            else:
                logger.info("Исходные позиции не найдены, используется автоматическое размещение")
                use_original_positions = False
        else:
            logger.info("Запуск оптимизации размещения (исходные позиции игнорируются)")
        
        self.agents = []
        
        num_shapes = len(shapes)
        if num_shapes > 0 and not use_original_positions:
            total_width = self.sheet_size[0] * 0.8  # Используем 80% ширины
            spacing = total_width / max(1, num_shapes - 1) if num_shapes > 1 else 0
            start_x = self.sheet_size[0] * 0.1  # Начинаем с 10% от левого края
            logger.info(f"Распределение по X: start_x={start_x:.2f} мм, spacing={spacing:.2f} мм, total_width={total_width:.2f} мм")
        
        for i, shape in enumerate(shapes):
            if use_original_positions and hasattr(shape, 'original_position'):
                init_position = np.array(shape.original_position)
                logger.info(f"  Использование исходной позиции для {shape.name}: ({init_position[0]:.2f}, {init_position[1]:.2f}) мм")
            else:
                if num_shapes > 1:
                    init_x = start_x + (i * spacing)
                else:
                    init_x = self.sheet_size[0] / 2.0
                
                shape_height = shape.get_bounding_box()[3] - shape.get_bounding_box()[1]
                init_y = self.sheet_size[1] * 0.6 + shape_height * 0.5
                
                init_position = np.array([init_x, init_y])
            
            # Установка приоритета
            priority = priorities[i] if priorities and i < len(priorities) else 1
            
            # Создание агента
            agent = IAGIAgent(
                shape=shape,
                position=init_position,
                angle=0.0,
                priority=priority
            )
            
            # Установка технологических ограничений
            agent.min_gap = self.min_gap
            agent.sheet_size = self.sheet_size  # Для проверки границ в dynamics
            if orientation_constraints and i < len(orientation_constraints):
                agent.orientation_constraints = orientation_constraints[i]
            
            bbox = shape.get_bounding_box()
            logger.info(f"  Агент {i+1}: {shape.name}, приоритет={priority}, "
                       f"начальная позиция=({init_position[0]:.2f}, {init_position[1]:.2f}) мм, "
                       f"размер=({bbox[2]-bbox[0]:.2f} x {bbox[3]-bbox[1]:.2f}) мм, "
                       f"площадь={shape.area:.2f} мм2, масса={agent.mass:.2f}")
            
            self.agents.append(agent)
        
        logger.info(f"Подготовка завершена: создано {len(self.agents)} агентов")
        return self.agents
    
    def optimize(self, shapes, priorities=None, orientation_constraints=None, use_original_positions=True, progress_callback=None):
        """
        Основной метод оптимизации раскроя
        
        Args:
            shapes: список фигур для размещения
            priorities: приоритеты фигур (опционально)
            orientation_constraints: ограничения ориентации (опционально)
            use_original_positions: использовать исходные позиции из файла
            progress_callback: функция обратного вызова для обновления прогресса (progress, agents, utilization)
        """
        start_time = time.time()
        
        # Подготовка агентов
        self.prepare_agents(shapes, priorities, orientation_constraints, use_original_positions)
        
        if use_original_positions:
            logger.info("Использование исходных позиций из DXF как начальных позиций для оптимизации")
        else:
            logger.info("Использование автоматического начального размещения для оптимизации")
        
        # Строим пространственный индекс
        self.collision_detector.build_spatial_index(
            [agent.shape for agent in self.agents], 
            self.sheet_size
        )
        
        # Выбор алгоритма размещения
        if self.use_sequential:
            self.result = sequential_placement(
                self.agents,
                self.sheet_size,
                self.collision_detector,
                self.dynamics,
                self.min_gap,
                self.defect_zones,
                time_limit=self.time_limit,
                progress_callback=progress_callback
            )
        else:
            self.result = parallel_placement(
                self.agents,
                self.sheet_size,
                self.collision_detector,
                self.dynamics,
                self.min_gap,
                self.defect_zones,
                time_limit=self.time_limit,
                progress_callback=progress_callback
            )
        
        # Применение механизмов стабилизации при необходимости
        if self.stabilization_enabled:
            logger.info("Применение механизмов стабилизации...")
            self.result = apply_stabilization(
                self.agents,  # agents
                self.collision_detector,  # collision_detector
                self.dynamics,  # dynamics
                self.min_gap  # min_gap
            )
            logger.info("Стабилизация завершена")
            
        elapsed_time = time.time() - start_time
        print(f"Оптимизация завершена за {elapsed_time:.2f} секунд")
        logger.info(f"Оптимизация завершена за {elapsed_time:.2f} секунд")
        
        # Расчет коэффициента использования материала
        total_area = sum(agent.shape.area for agent in self.agents)
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for agent in self.agents:
            transformed_shape = agent.get_transformed_shape()
            x_min, y_min, x_max, y_max = transformed_shape.get_bounding_box()
            min_x = min(min_x, x_min)
            min_y = min(min_y, y_min)
            max_x = max(max_x, x_max)
            max_y = max(max_y, y_max)
        if max_x > min_x and max_y > min_y:
            effective_area = (max_x - min_x) * (max_y - min_y)
        else:
            effective_area = self.sheet_size[0] * max_y if max_y > 0 else 1.0
        
        if effective_area < total_area:
            effective_area = total_area
        
        utilization = (total_area / effective_area) * 100 if effective_area > 0 else 0.0
        

        print(f"Коэффициент использования материала: {utilization:.2f}%")
        print(f"  Площадь фигур: {total_area:.2f} мм²")
        print(f"  Использованная высота: {max_y:.2f} мм")
        print(f"  Эффективная площадь: {effective_area:.2f} мм²")
        
        return self.result
    
    def optimize_batch(self,
                  parts: List[Any],
                  sheet_batch: SheetBatchManager,
                  priorities: Optional[List[int]] = None,
                  orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Оптимизация раскроя партии деталей по нескольким листам
        
        :param parts: Список деталей для раскроя
        :param sheet_batch: Менеджер партии листов
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Результаты оптимизации по всем листам
        """
        from algorithms.batch_coordinator import BatchNestingCoordinator
        
        coordinator = BatchNestingCoordinator(
            sheet_batch=sheet_batch,
            constraint_manager=self.constraint_manager,
            distribution_method='hybrid_ga',
            time_limit=self.time_limit
        )
        
        return coordinator.coordinate_batch_nesting(
            parts=parts,
            priorities=priorities,
            orientation_constraints=orientation_constraints
        )
    
    def get_final_positions(self):
        """
        Получение финальных позиций всех фигур
        """
        positions = []
        for agent in self.agents:
            transformed_shape = agent.get_transformed_shape()
            positions.append({
                'name': agent.shape.name,
                'position': agent.position.tolist(),
                'angle': agent.angle,
                'polygon': transformed_shape.polygon
            })
        return positions