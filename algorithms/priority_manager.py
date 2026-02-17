"""
Модуль управления приоритетами размещения
Согласно разделу 3.3.3 и 2.5.4, приоритеты критичны для технологической корректности
"""

from typing import List, Dict, Tuple, Any, Optional
from core.agent import IAGIAgent
from core.constraints import ConstraintManager
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PriorityManager:
    """
    Менеджер приоритетов для управления последовательностью размещения
    
    Реализует:
    - Назначение приоритетов на основе площади, сложности или вручную
    - Формирование уровней приоритетов
    - Управление последовательным размещением по приоритетам
    - Динамическую корректировку приоритетов
    """
    
    def __init__(self, constraint_manager: Optional[ConstraintManager] = None):
        """
        Инициализация менеджера приоритетов
        
        :param constraint_manager: Менеджер технологических ограничений
        """
        self.constraint_manager = constraint_manager
        self.priority_strategy = 'area'  # 'area', 'complexity', 'manual'
        self.max_priority_level = 5
    
    def assign_priorities(self, agents: List[IAGIAgent], 
                         strategy: str = 'area') -> List[IAGIAgent]:
        """
        Назначение приоритетов агентам на основе выбранной стратегии
        
        :param agents: Список агентов
        :param strategy: Стратегия назначения ('area', 'complexity', 'manual')
        :return: Список агентов с назначенными приоритетами
        """
        self.priority_strategy = strategy
        
        if strategy == 'area':
            return self._assign_by_area(agents)
        elif strategy == 'complexity':
            return self._assign_by_complexity(agents)
        elif strategy == 'manual':
            return self._assign_manual(agents)
        else:
            logger.warning(f"Неизвестная стратегия '{strategy}'. Используется 'area'.")
            return self._assign_by_area(agents)
    
    def _assign_by_area(self, agents: List[IAGIAgent]) -> List[IAGIAgent]:
        """Назначение приоритетов на основе площади фигур (чем больше площадь, тем выше приоритет)"""
        # Сортировка по площади в убывающем порядке
        sorted_agents = sorted(agents, key=lambda a: a.shape.area, reverse=True)
        
        # Распределение по уровням приоритетов
        num_levels = min(self.max_priority_level, len(agents))
        agents_per_level = max(1, len(agents) // num_levels)
        
        for i, agent in enumerate(sorted_agents):
            priority_level = min(i // agents_per_level + 1, self.max_priority_level)
            agent.priority = priority_level
            
            # Установка приоритета в менеджере ограничений (если доступен)
            if self.constraint_manager:
                shape_id = getattr(agent.shape, 'name', str(id(agent)))
                self.constraint_manager.set_placement_priority(shape_id, priority_level)
        
        logger.info(f"Приоритеты назначены по площади. Уровней: {num_levels}")
        return sorted_agents
    
    def _assign_by_complexity(self, agents: List[IAGIAgent]) -> List[IAGIAgent]:
        """Назначение приоритетов на основе сложности геометрии (число вершин, невыпуклость)"""
        # Расчет сложности для каждой фигуры
        complexity_scores = []
        
        for agent in agents:
            # Число вершин внешнего контура
            num_vertices = len(agent.shape.outer_contour)
            
            # Проверка на невыпуклость (наличие реентрантов)
            has_reentrants = self._has_reentrants(agent.shape)
            
            # Сложность = число вершин + штраф за невыпуклость
            complexity = num_vertices + (10 if has_reentrants else 0)
            complexity_scores.append((agent, complexity))
        
        # Сортировка по сложности в убывающем порядке
        sorted_agents = [agent for agent, _ in sorted(complexity_scores, key=lambda x: x[1], reverse=True)]
        
        # Распределение по уровням приоритетов
        num_levels = min(self.max_priority_level, len(agents))
        agents_per_level = max(1, len(agents) // num_levels)
        
        for i, agent in enumerate(sorted_agents):
            priority_level = min(i // agents_per_level + 1, self.max_priority_level)
            agent.priority = priority_level
            
            if self.constraint_manager:
                shape_id = getattr(agent.shape, 'name', str(id(agent)))
                self.constraint_manager.set_placement_priority(shape_id, priority_level)
        
        logger.info(f"Приоритеты назначены по сложности. Уровней: {num_levels}")
        return sorted_agents
    
    def _has_reentrants(self, shape: Any) -> bool:
        """Проверка наличия реентрантов (вогнутостей) в фигуре"""
        # Упрощенная проверка: если фигура не выпуклая
        try:
            return not shape.polygon.is_convex
        except:
            # Если метод is_convex недоступен, используем приближенную проверку
            coords = list(shape.polygon.exterior.coords)[:-1]
            if len(coords) < 4:
                return False
            
            # Проверка знака векторного произведения
            prev_cross = None
            for i in range(len(coords)):
                p1 = np.array(coords[i])
                p2 = np.array(coords[(i + 1) % len(coords)])
                p3 = np.array(coords[(i + 2) % len(coords)])
                
                v1 = p2 - p1
                v2 = p3 - p2
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                
                if cross != 0:
                    if prev_cross is None:
                        prev_cross = cross
                    elif prev_cross * cross < 0:
                        return True
            
            return False
    
    def _assign_manual(self, agents: List[IAGIAgent]) -> List[IAGIAgent]:
        """Назначение приоритетов вручную (на основе данных из менеджера ограничений)"""
        if not self.constraint_manager:
            logger.warning("Невозможно назначить приоритеты вручную: менеджер ограничений не задан")
            return self._assign_by_area(agents)
        
        for agent in agents:
            shape_id = getattr(agent.shape, 'name', str(id(agent)))
            priority = self.constraint_manager.get_shape_priority(shape_id)
            agent.priority = priority
        
        # Сортировка по приоритету
        sorted_agents = sorted(agents, key=lambda a: a.priority, reverse=True)
        
        logger.info("Приоритеты назначены вручную на основе данных менеджера ограничений")
        return sorted_agents
    
    def get_priority_groups(self, agents: List[IAGIAgent]) -> Dict[int, List[IAGIAgent]]:
        """
        Группировка агентов по уровням приоритетов
        
        :param agents: Список агентов
        :return: Словарь {уровень_приоритета: [агенты]}
        """
        groups = {}
        
        for agent in agents:
            priority = agent.priority
            if priority not in groups:
                groups[priority] = []
            groups[priority].append(agent)
        
        # Сортировка групп по приоритету
        sorted_groups = dict(sorted(groups.items(), reverse=True))
        
        logger.debug(f"Сформировано {len(sorted_groups)} групп приоритетов")
        return sorted_groups
    
    def get_next_priority_group(self, agents: List[IAGIAgent], 
                               current_priority: Optional[int] = None) -> Optional[List[IAGIAgent]]:
        """
        Получение следующей группы агентов для размещения
        
        :param agents: Список всех агентов
        :param current_priority: Текущий уровень приоритета
        :return: Следующая группа агентов или None если все размещены
        """
        groups = self.get_priority_groups(agents)
        if not groups:
            return None
        
        # Если текущий приоритет не задан, возвращаем группу с наивысшим приоритетом
        if current_priority is None:
            highest_priority = max(groups.keys())
            return groups[highest_priority]
        
        # Поиск следующего приоритета (меньшего)
        sorted_priorities = sorted(groups.keys(), reverse=True)
        for priority in sorted_priorities:
            if priority < current_priority:
                return groups[priority]
        
        # Больше нет групп с меньшим приоритетом
        return None
    
    def freeze_high_priority_agents(self, agents: List[IAGIAgent], 
                                    threshold_priority: int) -> List[IAGIAgent]:
        """
        Заморозка агентов с приоритетом выше заданного порога
        
        :param agents: Список агентов
        :param threshold_priority: Пороговый приоритет
        :return: Обновленный список агентов
        """
        frozen_count = 0
        
        for agent in agents:
            if agent.priority > threshold_priority and not agent.is_frozen:
                agent.is_frozen = True
                frozen_count += 1
        
        logger.info(f"Заморожено {frozen_count} агентов с приоритетом > {threshold_priority}")
        return agents
    
    def adjust_priorities_dynamically(self, agents: List[IAGIAgent], 
                                     performance_metrics: Dict[str, float]) -> List[IAGIAgent]:
        """
        Динамическая корректировка приоритетов на основе метрик производительности
        
        :param agents: Список агентов
        :param performance_metrics: Метрики производительности
        :return: Обновленный список агентов
        """
        # Пример динамической корректировки:
        # Если система застопорилась, повышаем приоритет "проблемных" агентов
        
        stagnation_detected = performance_metrics.get('stagnation', False)
        energy_change = performance_metrics.get('energy_change', 0.0)
        
        if stagnation_detected or abs(energy_change) < 1e-3:
            logger.info("Обнаружена стагнация. Корректировка приоритетов...")
            
            # Находим агентов с низкой скоростью (потенциально "застрявшие")
            slow_agents = [a for a in agents if not a.is_frozen and np.linalg.norm(a.velocity) < 0.1]
            
            for agent in slow_agents:
                # Повышаем приоритет на 1 (но не выше максимального)
                agent.priority = min(agent.priority + 1, self.max_priority_level)
            
            logger.info(f"Приоритет повышен для {len(slow_agents)} агентов")
        
        return agents
