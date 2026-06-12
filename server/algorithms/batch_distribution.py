
import numpy as np
import random
import time
from typing import List, Dict, Tuple, Optional, Any
from core.geometry import PolygonShape
from core.sheet_batch import Sheet, SheetBatchManager
from core.optimizer import PackingOptimizer
from core.constraints import ConstraintManager
from algorithms.hybrid import HybridGeneticIAGI, HybridPSOIAGI
import logging

"""
Модуль для распределения партии деталей по листам материалов
Согласно разделу 3.4.1, гибридизация с глобальными метаэвристиками критична для оптимизации распределения
"""

logger = logging.getLogger(__name__)

class BatchDistributionOptimizer:
    """
    Оптимизатор распределения партии деталей по листам материалов
    
    Решает задачу бин-пакинга с учетом:
    - Размеров и характеристик листов
    - Геометрии деталей
    - Технологических ограничений
    - Минимизации числа используемых листов
    - Максимизации общего коэффициента использования материала
    """
    
    def __init__(self, 
                 sheet_batch: SheetBatchManager,
                 constraint_manager: ConstraintManager,
                 optimization_method: str = 'hybrid_ga',
                 time_limit: float = 300.0):
        """
        Инициализация оптимизатора распределения
        
        :param sheet_batch: Менеджер партии листов
        :param constraint_manager: Менеджер технологических ограничений
        :param optimization_method: Метод оптимизации ('first_fit', 'best_fit', 'hybrid_ga', 'hybrid_pso')
        :param time_limit: Лимит времени на оптимизацию в секундах
        """
        self.sheet_batch = sheet_batch
        self.constraint_manager = constraint_manager
        self.optimization_method = optimization_method
        self.time_limit = time_limit
        
        # Статистика оптимизации
        self.optimization_stats = {
            'total_sheets_used': 0,
            'total_utilization': 0.0,
            'optimization_time': 0.0,
            'distribution_method': optimization_method
        }
    
    def distribute_parts(self, 
                        parts: List[PolygonShape],
                        priorities: Optional[List[int]] = None,
                        orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Распределение деталей по листам
        
        :param parts: Список деталей для распределения
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Словарь с результатами распределения
        """
        start_time = time.time()
        logger.info(f"Начало распределения {len(parts)} деталей по листам...")
        
        # Выбор метода оптимизации
        if self.optimization_method == 'first_fit':
            distribution = self._first_fit_distribution(parts, priorities, orientation_constraints)
        elif self.optimization_method == 'best_fit':
            distribution = self._best_fit_distribution(parts, priorities, orientation_constraints)
        elif self.optimization_method == 'hybrid_ga':
            distribution = self._hybrid_ga_distribution(parts, priorities, orientation_constraints)
        elif self.optimization_method == 'hybrid_pso':
            distribution = self._hybrid_pso_distribution(parts, priorities, orientation_constraints)
        else:
            logger.warning(f"Неизвестный метод оптимизации '{self.optimization_method}'. Используется first_fit.")
            distribution = self._first_fit_distribution(parts, priorities, orientation_constraints)
        
        # Расчет статистики
        optimization_time = time.time() - start_time
        self.optimization_stats['optimization_time'] = optimization_time
        self.optimization_stats['total_sheets_used'] = len(distribution)
        
        total_utilization = sum(sheet_data['utilization'] for sheet_data in distribution.values())
        self.optimization_stats['total_utilization'] = total_utilization / len(distribution) if distribution else 0.0
        
        logger.info(f"Распределение завершено за {optimization_time:.2f} секунд")
        logger.info(f"Использовано листов: {len(distribution)}")
        logger.info(f"Средняя утилизация: {self.optimization_stats['total_utilization']:.2f}%")
        
        return {
            'distribution': distribution,
            'stats': self.optimization_stats,
            'optimization_time': optimization_time
        }
    
    def _first_fit_distribution(self, 
                               parts: List[PolygonShape],
                               priorities: Optional[List[int]] = None,
                               orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Алгоритм First-Fit для распределения деталей по листам
        
        :param parts: Список деталей
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Распределение деталей по листам
        """
        # Сортировка деталей по убыванию площади (для улучшения результата)
        sorted_indices = sorted(range(len(parts)), key=lambda i: parts[i].area, reverse=True)
        sorted_parts = [parts[i] for i in sorted_indices]
        
        if priorities:
            sorted_priorities = [priorities[i] for i in sorted_indices]
        else:
            sorted_priorities = None
        
        if orientation_constraints:
            sorted_orientation_constraints = [orientation_constraints[i] for i in sorted_indices]
        else:
            sorted_orientation_constraints = None
        
        # Инициализация распределения
        distribution = {}
        available_sheets = self.sheet_batch.get_available_sheets()
        
        if not available_sheets:
            raise ValueError("Нет доступных листов для распределения деталей")
        
        # Распределение деталей
        current_sheet_idx = 0
        current_sheet = available_sheets[current_sheet_idx]
        
        for i, part in enumerate(sorted_parts):
            # Проверка, помещается ли деталь на текущем листе
            if self._can_place_part_on_sheet(part, current_sheet):
                # Добавление детали в распределение для текущего листа
                if current_sheet.sheet_id not in distribution:
                    distribution[current_sheet.sheet_id] = {
                        'sheet': current_sheet,
                        'parts': [],
                        'utilization': 0.0
                    }
                
                distribution[current_sheet.sheet_id]['parts'].append({
                    'part': part,
                    'priority': sorted_priorities[i] if sorted_priorities else 1,
                    'orientation_constraint': sorted_orientation_constraints[i] if sorted_orientation_constraints else None
                })
            else:
                # Переход к следующему листу
                current_sheet_idx += 1
                if current_sheet_idx >= len(available_sheets):
                    logger.warning(f"Недостаточно листов для распределения всех деталей. Обработано {i} из {len(parts)} деталей.")
                    break
                
                current_sheet = available_sheets[current_sheet_idx]
                
                # Добавление детали в распределение для нового листа
                if current_sheet.sheet_id not in distribution:
                    distribution[current_sheet.sheet_id] = {
                        'sheet': current_sheet,
                        'parts': [],
                        'utilization': 0.0
                    }
                
                distribution[current_sheet.sheet_id]['parts'].append({
                    'part': part,
                    'priority': sorted_priorities[i] if sorted_priorities else 1,
                    'orientation_constraint': sorted_orientation_constraints[i] if sorted_orientation_constraints else None
                })
        
        # Оптимизация раскроя для каждого листа
        for sheet_id, sheet_data in distribution.items():
            sheet = sheet_data['sheet']
            parts_for_sheet = [item['part'] for item in sheet_data['parts']]
            priorities_for_sheet = [item['priority'] for item in sheet_data['parts']]
            orientation_constraints_for_sheet = [item['orientation_constraint'] for item in sheet_data['parts']]
            
            # Создание оптимизатора для листа
            optimizer = PackingOptimizer({
                'sheet_size': sheet.get_sheet_size(),
                'min_gap': sheet.min_gap,
                'time_limit': 60.0,
                'use_sequential': True,
                'stabilization_enabled': True
            })
            
            # Добавление дефектных зон
            for defect in sheet.defects:
                optimizer.add_defect_zone(defect)
            
            # Запуск оптимизации раскроя
            try:
                result_agents = optimizer.optimize(
                    shapes=parts_for_sheet,
                    priorities=priorities_for_sheet,
                    orientation_constraints=orientation_constraints_for_sheet
                )
                
                # Расчет утилизации
                total_area = sum(agent.shape.area for agent in result_agents)
                sheet_area = sheet.width * sheet.height
                utilization = (total_area / sheet_area) * 100.0
                
                sheet_data['utilization'] = utilization
                sheet_data['agents'] = result_agents
                
                logger.debug(f"Лист {sheet_id}: утилизация {utilization:.2f}%")
                
            except Exception as e:
                logger.error(f"Ошибка при оптимизации раскроя листа {sheet_id}: {e}")
                sheet_data['utilization'] = 0.0
                sheet_data['agents'] = []
        
        return distribution
    
    def _best_fit_distribution(self, 
                               parts: List[PolygonShape],
                               priorities: Optional[List[int]] = None,
                               orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Алгоритм Best-Fit для распределения деталей по листам
        
        :param parts: Список деталей
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Распределение деталей по листам
        """
        # Сортировка деталей по убыванию площади
        sorted_indices = sorted(range(len(parts)), key=lambda i: parts[i].area, reverse=True)
        sorted_parts = [parts[i] for i in sorted_indices]
        
        if priorities:
            sorted_priorities = [priorities[i] for i in sorted_indices]
        else:
            sorted_priorities = None
        
        if orientation_constraints:
            sorted_orientation_constraints = [orientation_constraints[i] for i in sorted_indices]
        else:
            sorted_orientation_constraints = None
        
        # Инициализация распределения
        distribution = {}
        available_sheets = self.sheet_batch.get_available_sheets()
        
        if not available_sheets:
            raise ValueError("Нет доступных листов для распределения деталей")
        
        # Распределение деталей
        for i, part in enumerate(sorted_parts):
            best_sheet = None
            best_utilization = -1.0
            
            # Поиск листа с наилучшей утилизацией для текущей детали
            for sheet in available_sheets:
                if not sheet.is_used and self._can_place_part_on_sheet(part, sheet):
                    # Временная оценка утилизации
                    temp_parts = [item['part'] for item in distribution.get(sheet.sheet_id, {}).get('parts', [])] + [part]
                    temp_area = sum(p.area for p in temp_parts)
                    temp_utilization = (temp_area / (sheet.width * sheet.height)) * 100.0
                    
                    if temp_utilization > best_utilization:
                        best_utilization = temp_utilization
                        best_sheet = sheet
            
            # Если не найдено подходящего листа, используем новый
            if best_sheet is None:
                for sheet in available_sheets:
                    if not sheet.is_used:
                        best_sheet = sheet
                        break
            
            if best_sheet is None:
                logger.warning(f"Недостаточно листов для распределения всех деталей. Обработано {i} из {len(parts)} деталей.")
                break
            
            # Добавление детали в распределение
            if best_sheet.sheet_id not in distribution:
                distribution[best_sheet.sheet_id] = {
                    'sheet': best_sheet,
                    'parts': [],
                    'utilization': 0.0
                }
            
            distribution[best_sheet.sheet_id]['parts'].append({
                'part': part,
                'priority': sorted_priorities[i] if sorted_priorities else 1,
                'orientation_constraint': sorted_orientation_constraints[i] if sorted_orientation_constraints else None
            })
        
        # Оптимизация раскроя для каждого листа (аналогично first_fit)
        for sheet_id, sheet_data in distribution.items():
            sheet = sheet_data['sheet']
            parts_for_sheet = [item['part'] for item in sheet_data['parts']]
            priorities_for_sheet = [item['priority'] for item in sheet_data['parts']]
            orientation_constraints_for_sheet = [item['orientation_constraint'] for item in sheet_data['parts']]
            
            optimizer = PackingOptimizer({
                'sheet_size': sheet.get_sheet_size(),
                'min_gap': sheet.min_gap,
                'time_limit': 60.0,
                'use_sequential': True,
                'stabilization_enabled': True
            })
            
            for defect in sheet.defects:
                optimizer.add_defect_zone(defect)
            
            try:
                result_agents = optimizer.optimize(
                    shapes=parts_for_sheet,
                    priorities=priorities_for_sheet,
                    orientation_constraints=orientation_constraints_for_sheet
                )
                
                total_area = sum(agent.shape.area for agent in result_agents)
                sheet_area = sheet.width * sheet.height
                utilization = (total_area / sheet_area) * 100.0
                
                sheet_data['utilization'] = utilization
                sheet_data['agents'] = result_agents
                
            except Exception as e:
                logger.error(f"Ошибка при оптимизации раскроя листа {sheet_id}: {e}")
                sheet_data['utilization'] = 0.0
                sheet_data['agents'] = []
        
        return distribution
    
    def _hybrid_ga_distribution(self, 
                               parts: List[PolygonShape],
                               priorities: Optional[List[int]] = None,
                               orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Гибридный генетический алгоритм для распределения деталей по листам
        
        :param parts: Список деталей
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Распределение деталей по листам
        """
        # Создание гибридного оптимизатора ГА-ИАГИ
        available_sheets = self.sheet_batch.get_available_sheets()
        
        if not available_sheets:
            raise ValueError("Нет доступных листов для распределения деталей")
        
        # Определение максимального размера листа для нормализации
        max_sheet_size = max((sheet.width * sheet.height) for sheet in available_sheets)
        
        # Создание оптимизатора
        hybrid_optimizer = HybridGeneticIAGI(
            shapes=parts,
            sheet_size=(max_sheet_size**0.5, max_sheet_size**0.5),  # Квадратный лист максимального размера
            min_gap=self.constraint_manager.current_gap,
            population_size=30,
            generations=50,
            time_limit=self.time_limit / 2  # Половина времени для ГА
        )
        
        # Запуск оптимизации
        try:
            best_solution = hybrid_optimizer.optimize()
            
            # Декодирование решения в распределение по листам
            distribution = self._decode_ga_solution_to_distribution(
                best_solution, 
                parts, 
                available_sheets,
                priorities,
                orientation_constraints
            )
            
            return distribution
            
        except Exception as e:
            logger.error(f"Ошибка при гибридной ГА оптимизации: {e}")
            # Возврат к методу first_fit в случае ошибки
            return self._first_fit_distribution(parts, priorities, orientation_constraints)
    
    def _hybrid_pso_distribution(self, 
                                parts: List[PolygonShape],
                                priorities: Optional[List[int]] = None,
                                orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Гибридный алгоритм роя частиц для распределения деталей по листам
        
        :param parts: Список деталей
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Распределение деталей по листам
        """
        # Аналогично гибридному ГА, но с использованием PSO
        available_sheets = self.sheet_batch.get_available_sheets()
        
        if not available_sheets:
            raise ValueError("Нет доступных листов для распределения деталей")
        
        max_sheet_size = max((sheet.width * sheet.height) for sheet in available_sheets)
        
        # Создание оптимизатора PSO-ИАГИ
        from algorithms.hybrid import HybridPSOIAGI
        
        hybrid_optimizer = HybridPSOIAGI(
            shapes=parts,
            sheet_size=(max_sheet_size**0.5, max_sheet_size**0.5),
            min_gap=self.constraint_manager.current_gap,
            swarm_size=30,
            max_iterations=50,
            time_limit=self.time_limit / 2
        )
        
        try:
            best_solution = hybrid_optimizer.optimize()
            
            distribution = self._decode_pso_solution_to_distribution(
                best_solution, 
                parts, 
                available_sheets,
                priorities,
                orientation_constraints
            )
            
            return distribution
            
        except Exception as e:
            logger.error(f"Ошибка при гибридной PSO оптимизации: {e}")
            return self._first_fit_distribution(parts, priorities, orientation_constraints)
    
    def _decode_ga_solution_to_distribution(self,
                                           solution: Any,
                                           parts: List[PolygonShape],
                                           sheets: List[Sheet],
                                           priorities: Optional[List[int]] = None,
                                           orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Декодирование решения ГА в распределение по листам
        
        :param solution: Решение от гибридного оптимизатора
        :param parts: Список деталей
        :param sheets: Список доступных листов
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Распределение деталей по листам
        """
        # В реальной реализации здесь будет сложная логика декодирования
        # Для упрощения используем эвристику: распределяем детали по листам в порядке их размещения
        
        distribution = {}
        current_sheet_idx = 0
        
        # Предполагаем, что решение содержит отсортированный список деталей
        if hasattr(solution, 'placement') and solution.placement:
            sorted_parts = [item['shape'] for item in solution.placement]
        else:
            sorted_parts = parts
        
        for i, part in enumerate(sorted_parts):
            if current_sheet_idx >= len(sheets):
                logger.warning(f"Недостаточно листов. Обработано {i} из {len(parts)} деталей.")
                break
            
            sheet = sheets[current_sheet_idx]
            
            if sheet.sheet_id not in distribution:
                distribution[sheet.sheet_id] = {
                    'sheet': sheet,
                    'parts': [],
                    'utilization': 0.0
                }
            
            # Добавление детали
            distribution[sheet.sheet_id]['parts'].append({
                'part': part,
                'priority': priorities[i] if priorities and i < len(priorities) else 1,
                'orientation_constraint': orientation_constraints[i] if orientation_constraints and i < len(orientation_constraints) else None
            })
            
            # Эвристическая проверка заполненности листа
            total_area = sum(p['part'].area for p in distribution[sheet.sheet_id]['parts'])
            sheet_area = sheet.width * sheet.height
            
            if total_area / sheet_area > 0.85:  # Если лист заполнен на 85%
                # Оптимизация раскроя для текущего листа
                self._optimize_sheet_cutting(
                    distribution[sheet.sheet_id],
                    sheet
                )
                
                current_sheet_idx += 1
        
        # Оптимизация оставшихся листов
        for sheet_id, sheet_data in distribution.items():
            if sheet_data['utilization'] == 0.0:
                self._optimize_sheet_cutting(sheet_data, sheet_data['sheet'])
        
        return distribution
    
    def _decode_pso_solution_to_distribution(self,
                                            solution: Any,
                                            parts: List[PolygonShape],
                                            sheets: List[Sheet],
                                            priorities: Optional[List[int]] = None,
                                            orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Декодирование решения PSO в распределение по листам (аналогично ГА)"""
        return self._decode_ga_solution_to_distribution(
            solution, parts, sheets, priorities, orientation_constraints
        )
    
    def _optimize_sheet_cutting(self, sheet_data: Dict[str, Any], sheet: Sheet):
        """Оптимизация раскроя для конкретного листа
        
        :param sheet_data: Данные о листе из распределения
        :param sheet: Объект листа
        """
        parts_for_sheet = [item['part'] for item in sheet_data['parts']]
        priorities_for_sheet = [item['priority'] for item in sheet_data['parts']]
        orientation_constraints_for_sheet = [item['orientation_constraint'] for item in sheet_data['parts']]
        
        optimizer = PackingOptimizer({
            'sheet_size': sheet.get_sheet_size(),
            'min_gap': sheet.min_gap,
            'time_limit': 60.0,
            'use_sequential': True,
            'stabilization_enabled': True
        })
        
        for defect in sheet.defects:
            optimizer.add_defect_zone(defect)
        
        try:
            result_agents = optimizer.optimize(
                shapes=parts_for_sheet,
                priorities=priorities_for_sheet,
                orientation_constraints=orientation_constraints_for_sheet
            )
            
            total_area = sum(agent.shape.area for agent in result_agents)
            sheet_area = sheet.width * sheet.height
            utilization = (total_area / sheet_area) * 100.0
            
            sheet_data['utilization'] = utilization
            sheet_data['agents'] = result_agents
            
        except Exception as e:
            logger.error(f"Ошибка при оптимизации раскроя листа: {e}")
            sheet_data['utilization'] = 0.0
            sheet_data['agents'] = []
    
    def _can_place_part_on_sheet(self, part: PolygonShape, sheet: Sheet) -> bool:
        """
        Проверка возможности размещения детали на листе
        
        :param part: Деталь
        :param sheet: Лист
        :return: True если деталь может быть размещена на листе
        """
        # Простая проверка по площади
        part_area = part.area
        sheet_area = sheet.width * sheet.height
        used_area = sheet_area * sheet.utilization / 100.0 if sheet.is_used else 0.0
        remaining_area = sheet_area - used_area
        
        # Учитываем минимальный зазор и дефектные зоны
        min_gap_area = sheet.min_gap * (sheet.width + sheet.height) * 2
        defects_area = sum(defect.area for defect in sheet.defects)
        
        effective_remaining_area = remaining_area - min_gap_area - defects_area
        
        return part_area <= effective_remaining_area * 0.9  # 10% запас для зазоров
    
    def get_distribution_summary(self, distribution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Получение сводки по распределению
        
        :param distribution: Результат распределения
        :return: Словарь со сводкой
        """
        total_sheets = len(distribution)
        total_parts = sum(len(data['parts']) for data in distribution.values())
        total_utilization = sum(data['utilization'] for data in distribution.values())
        avg_utilization = total_utilization / total_sheets if total_sheets > 0 else 0.0
        
        sheets_by_utilization = sorted(
            [(sheet_id, data['utilization']) for sheet_id, data in distribution.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_sheets': total_sheets,
            'total_parts': total_parts,
            'avg_utilization': avg_utilization,
            'min_utilization': min(data['utilization'] for data in distribution.values()) if distribution else 0.0,
            'max_utilization': max(data['utilization'] for data in distribution.values()) if distribution else 0.0,
            'sheets_by_utilization': sheets_by_utilization
        }
