
"""
Модуль многозапусковой стратегии для выхода из локальных минимумов
Согласно разделу 3.5.4, многозапусковая стратегия критична для глобальной оптимизации
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from core.agent import IAGIAgent
from core.optimizer import PackingOptimizer
from core.constraints import ConstraintManager
import logging

logger = logging.getLogger(__name__)

class MultiStartOptimizer:
    """
    Оптимизатор с многозапусковой стратегией
    
    Реализует:
    - Несколько независимых запусков с разными начальными условиями
    - Выбор лучшего решения по критерию плотности упаковки
    - Адаптивное управление числом запусков
    - Параллельное выполнение запусков (опционально)
    """
    
    def __init__(self, base_optimizer: PackingOptimizer, 
                 num_starts: int = 5,
                 time_limit_per_start: float = 60.0,
                 parallel: bool = False):
        """
        Инициализация многозапускового оптимизатора
        
        :param base_optimizer: Базовый оптимизатор для каждого запуска
        :param num_starts: Число запусков
        :param time_limit_per_start: Лимит времени на один запуск (сек)
        :param parallel: Использовать параллельные вычисления
        """
        self.base_optimizer = base_optimizer
        self.num_starts = num_starts
        self.time_limit_per_start = time_limit_per_start
        self.parallel = parallel
        self.best_solution = None
        self.best_utilization = 0.0
        self.all_results = []
    
    def optimize(self, shapes: List[Any], 
                priorities: Optional[List[int]] = None,
                orientation_constraints: Optional[List[Any]] = None) -> List[IAGIAgent]:
        """
        Запуск многозапусковой оптимизации
        
        :param shapes: Список фигур для размещения
        :param priorities: Приоритеты фигур
        :param orientation_constraints: Ограничения на ориентацию
        :return: Лучшее найденное решение
        """
        logger.info(f"Запуск многозапусковой оптимизации: {self.num_starts} запусков")
        
        start_time = time.time()
        
        if self.parallel:
            results = self._optimize_parallel(shapes, priorities, orientation_constraints)
        else:
            results = self._optimize_sequential(shapes, priorities, orientation_constraints)
        
        # Выбор лучшего решения
        best_result = max(results, key=lambda r: r['utilization'])
        
        self.best_solution = best_result['agents']
        self.best_utilization = best_result['utilization']
        self.all_results = results
        
        total_time = time.time() - start_time
        logger.info(f"Многозапусковая оптимизация завершена за {total_time:.2f} секунд")
        logger.info(f"Лучший результат: {self.best_utilization:.2f}% использования материала")
        
        return self.best_solution
    
    def _optimize_sequential(self, shapes: List[Any], 
                            priorities: Optional[List[int]] = None,
                            orientation_constraints: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Последовательное выполнение запусков"""
        results = []
        
        for start_num in range(self.num_starts):
            logger.info(f"Запуск {start_num + 1}/{self.num_starts}")
            
            # Создание копии оптимизатора для независимого запуска
            optimizer_copy = self._copy_optimizer()
            
            # Установка уникального случайного зерна для каждого запуска
            np.random.seed(int(time.time() * 1000) % (2**32) + start_num)
            
            # Запуск оптимизации
            start_time = time.time()
            agents = optimizer_copy.optimize(
                shapes=shapes,
                priorities=priorities,
                orientation_constraints=orientation_constraints
            )
            elapsed_time = time.time() - start_time
            
            # Расчет метрик
            utilization = self._calculate_utilization(agents)
            
            results.append({
                'start_num': start_num + 1,
                'agents': agents,
                'utilization': utilization,
                'time': elapsed_time
            })
            
            logger.info(f"Запуск {start_num + 1} завершен: {utilization:.2f}% за {elapsed_time:.2f} сек")
        
        return results
    
    def _optimize_parallel(self, shapes: List[Any], 
                          priorities: Optional[List[int]] = None,
                          orientation_constraints: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Параллельное выполнение запусков"""
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            results = []
            
            # Функция для одного запуска
            def single_start(start_num):
                # Создание копии оптимизатора
                optimizer_copy = self._copy_optimizer()
                
                # Установка уникального случайного зерна
                np.random.seed(int(time.time() * 1000) % (2**32) + start_num)
                # Запуск оптимизации
                start_time = time.time()
                agents = optimizer_copy.optimize(
                    shapes=shapes,
                    priorities=priorities,
                    orientation_constraints=orientation_constraints
                )
                elapsed_time = time.time() - start_time
                
                # Расчет метрик
                utilization = self._calculate_utilization(agents)
                
                return {
                    'start_num': start_num + 1,
                    'agents': agents,
                    'utilization': utilization,
                    'time': elapsed_time
                }
            
            # Параллельное выполнение
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(single_start, i) for i in range(self.num_starts)]
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    logger.info(f"Запуск {result['start_num']} завершен: {result['utilization']:.2f}%")
            
            return results
            
        except ImportError as e:
            logger.warning(f"Не удалось импортировать concurrent.futures: {e}. Переключение на последовательный режим.")
            return self._optimize_sequential(shapes, priorities, orientation_constraints)
    
    def _copy_optimizer(self) -> PackingOptimizer:
        """Создание копии базового оптимизатора"""
        # Создание нового оптимизатора с такими же параметрами
        config = {
            'sheet_size': self.base_optimizer.sheet_size,
            'min_gap': self.base_optimizer.min_gap,
            'time_limit': self.time_limit_per_start,
            'use_sequential': self.base_optimizer.use_sequential,
            'stabilization_enabled': self.base_optimizer.stabilization_enabled
        }
        
        optimizer_copy = PackingOptimizer(config)
        
        # Копирование дефектных зон
        for defect in self.base_optimizer.defect_zones:
            optimizer_copy.add_defect_zone(defect)
        
        return optimizer_copy
    
    def _calculate_utilization(self, agents: List[IAGIAgent]) -> float:
        """Расчет коэффициента использования материала"""
        total_area = sum(agent.shape.area for agent in agents)
        sheet_area = self.base_optimizer.sheet_size[0] * self.base_optimizer.sheet_size[1]
        return (total_area / sheet_area) * 100.0
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Получение сводки по результатам всех запусков"""
        if not self.all_results:
            return {}
        
        utilizations = [r['utilization'] for r in self.all_results]
        
        return {
            'best_utilization': self.best_utilization,
            'average_utilization': np.mean(utilizations),
            'std_utilization': np.std(utilizations),
            'min_utilization': np.min(utilizations),
            'max_utilization': np.max(utilizations),
            'num_starts': len(self.all_results),
            'total_time': sum(r['time'] for r in self.all_results)
        }
    
    def export_results_summary(self, filename: str = "multi_start_summary.txt"):
        """Экспорт сводки результатов в файл"""
        summary = self.get_results_summary()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== СВОДКА РЕЗУЛЬТАТОВ МНОГОЗАПУСКОВОЙ ОПТИМИЗАЦИИ ===\n\n")
            f.write(f"Число запусков: {summary['num_starts']}\n")
            f.write(f"Лучший результат: {summary['best_utilization']:.2f}%\n")
            f.write(f"Средний результат: {summary['average_utilization']:.2f}%\n")
            f.write(f"Стандартное отклонение: {summary['std_utilization']:.2f}%\n")
            f.write(f"Минимальный результат: {summary['min_utilization']:.2f}%\n")
            f.write(f"Максимальный результат: {summary['max_utilization']:.2f}%\n")
            f.write(f"Общее время: {summary['total_time']:.2f} сек\n")
        
        logger.info(f"Сводка результатов сохранена в {filename}")

def multi_start_optimization(optimizer: PackingOptimizer,
                            shapes: List[Any],
                            num_starts: int = 5,
                            time_limit_per_start: float = 60.0,
                            parallel: bool = False) -> List[IAGIAgent]:
    """
    Функция-обертка для многозапусковой оптимизации
    
    :param optimizer: Базовый оптимизатор
    :param shapes: Список фигур
    :param num_starts: Число запусков
    :param time_limit_per_start: Лимит времени на запуск
    :param parallel: Использовать параллельные вычисления
    :return: Лучшее найденное решение
    """
    multi_start = MultiStartOptimizer(
        base_optimizer=optimizer,
        num_starts=num_starts,
        time_limit_per_start=time_limit_per_start,
        parallel=parallel
    )
    
    return multi_start.optimize(shapes)