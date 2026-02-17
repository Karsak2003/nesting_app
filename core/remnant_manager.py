
"""
Модуль учета остатков листовых материалов
Согласно разделу 2.5.5, учет остатков критичен для минимизации отходов в промышленном производстве
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from core.geometry import PolygonShape
from core.sheet_batch import Sheet
import logging

logger = logging.getLogger(__name__)

class Remnant:
    """
    Класс для представления остатка листового материала
    
    Остаток - это часть листа, которая осталась после раскроя и может быть использована для будущих деталей
    """
    
    def __init__(self,
                 remnant_id: str,
                 sheet_id: str,
                 contour: List[Tuple[float, float]],
                 area: float,
                 bounding_box: Tuple[float, float, float, float],
                 material_type: str = "steel"):
        """
        Инициализация остатка
        
        :param remnant_id: Уникальный идентификатор остатка
        :param sheet_id: Идентификатор исходного листа
        :param contour: Контур остатка
        :param area: Площадь остатка
        :param bounding_box: Ограничивающий прямоугольник (min_x, min_y, max_x, max_y)
        :param material_type: Тип материала
        """
        self.remnant_id = remnant_id
        self.sheet_id = sheet_id
        self.contour = contour
        self.area = area
        self.bounding_box = bounding_box
        self.material_type = material_type
        
        # Статус остатка
        self.is_available = True
        self.used_for_parts = []
        
        # Геометрическое представление
        self.polygon = PolygonShape(contour, name=f"remnant_{remnant_id}")
    
    def can_fit_part(self, part: PolygonShape, min_gap: float = 1.0) -> bool:
        """
        Проверка возможности размещения детали в остатке
        
        :param part: Деталь для размещения
        :param min_gap: Минимальный зазор
        :return: True если деталь помещается в остаток
        """
        # Простая проверка по площади и ограничивающему прямоугольнику
        if part.area > self.area * 0.9:  # 10% запас для зазоров
            return False
        
        part_bbox = part.get_bounding_box()
        remnant_bbox = self.bounding_box
        
        part_width = part_bbox[2] - part_bbox[0]
        part_height = part_bbox[3] - part_bbox[1]
        remnant_width = remnant_bbox[2] - remnant_bbox[0]
        remnant_height = remnant_bbox[3] - remnant_bbox[1]
        
        if part_width > remnant_width or part_height > remnant_height:
            return False
        
        # Более точная проверка через геометрическое пересечение
        # (в реальной реализации потребуется полная проверка)
        return True
    
    def mark_as_used(self, part_id: str):
        """Пометка остатка как использованного для детали"""
        self.used_for_parts.append(part_id)
        logger.debug(f"Остаток {self.remnant_id} использован для детали {part_id}")
    
    def get_effective_area(self) -> float:
        """Получение эффективной площади остатка (с учетом зазоров)"""
        # Учитываем 10% площади для технологических зазоров
        return self.area * 0.9
    def __str__(self):
        return (f"Remnant(id={self.remnant_id}, sheet={self.sheet_id}, "
                f"area={self.area:.2f}mm², available={self.is_available})")

class RemnantManager:
    """
    Менеджер остатков листовых материалов
    
    Обеспечивает:
    - Учет остатков после раскроя листов
    - Поиск подходящих остатков для новых деталей
    - Оптимизацию использования остатков
    - Отчетность по остаткам
    """
    
    def __init__(self):
        """Инициализация менеджера остатков"""
        self.remnants: List[Remnant] = []
        self.remnant_index: Dict[str, Remnant] = {}
    
    def add_remnant_from_sheet(self, sheet: Sheet, placed_parts: List[Any]):
        """
        Добавление остатков от листа после раскроя
        
        :param sheet: Обработанный лист
        :param placed_parts: Размещенные детали
        """
        # Расчет оставшейся площади
        total_part_area = sum(part.area for part in placed_parts if hasattr(part, 'area'))
        remaining_area = sheet.width * sheet.height - total_part_area
        
        if remaining_area < sheet.width * sheet.height * 0.05:  # Менее 5% - не учитываем
            logger.debug(f"Остаток листа {sheet.sheet_id} слишком мал ({remaining_area:.2f}mm²), не учитывается")
            return
        
        # Создание остатка (упрощенная реализация)
        # В реальной системе потребуется анализ свободного пространства
        remnant_id = f"remnant_{sheet.sheet_id}_{len(self.remnants)+1}"
        
        # Упрощенное представление остатка как прямоугольника
        # В реальной реализации нужно анализировать контур свободного пространства
        utilization_ratio = sheet.utilization / 100.0
        remnant_width = sheet.width * (1 - utilization_ratio**0.5)
        remnant_height = sheet.height * (1 - utilization_ratio**0.5)
        
        contour = [
            (sheet.width - remnant_width, sheet.height - remnant_height),
            (sheet.width, sheet.height - remnant_height),
            (sheet.width, sheet.height),
            (sheet.width - remnant_width, sheet.height)
        ]
        
        bounding_box = (
            sheet.width - remnant_width,
            sheet.height - remnant_height,
            sheet.width,
            sheet.height
        )
        
        remnant = Remnant(
            remnant_id=remnant_id,
            sheet_id=sheet.sheet_id,
            contour=contour,
            area=remaining_area,
            bounding_box=bounding_box,
            material_type=sheet.material_type
        )
        
        self.add_remnant(remnant)
        logger.info(f"Добавлен остаток {remnant_id} от листа {sheet.sheet_id}: {remaining_area:.2f}mm²")
    
    def add_remnant(self, remnant: Remnant):
        """Добавление остатка в менеджер"""
        self.remnants.append(remnant)
        self.remnant_index[remnant.remnant_id] = remnant
    
    def find_suitable_remnants(self, 
                               part: PolygonShape, 
                               material_type: Optional[str] = None,
                               min_gap: float = 1.0) -> List[Remnant]:
        """
        Поиск подходящих остатков для размещения детали
        
        :param part: Деталь для размещения
        :param material_type: Тип материала (опционально)
        :param min_gap: Минимальный зазор
        :return: Список подходящих остатков, отсортированных по площади
        """
        suitable = []
        
        for remnant in self.remnants:
            if not remnant.is_available:
                continue
            
            if material_type and remnant.material_type != material_type:
                continue
            
            if remnant.can_fit_part(part, min_gap):
                suitable.append(remnant)
        
        # Сортировка по площади (от меньшего к большему для минимизации отходов)
        suitable.sort(key=lambda r: r.area)
        
        return suitable
    
    def allocate_remnant_for_part(self, part:PolygonShape, part_id: str) -> Optional[Remnant]:
        """
        Выделение остатка для детали
        
        :param part: Деталь для размещения
        :param part_id: Идентификатор детали
        :return: Выделенный остаток или None
        """
        suitable = self.find_suitable_remnants(part)
        
        if not suitable:
            return None
        
        # Выбираем наименьший подходящий остаток
        remnant = suitable[0]
        remnant.is_available = False
        remnant.mark_as_used(part_id)
        
        logger.info(f"Остаток {remnant.remnant_id} выделен для детали {part_id}")
        return remnant
    
    def get_remnants_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по остаткам
        
        :return: Словарь со статистикой
        """
        total_remnants = len(self.remnants)
        available_remnants = sum(1 for r in self.remnants if r.is_available)
        used_remnants = total_remnants - available_remnants
        
        total_area = sum(r.area for r in self.remnants)
        available_area = sum(r.area for r in self.remnants if r.is_available)
        used_area = total_area - available_area
        
        avg_area = total_area / total_remnants if total_remnants > 0 else 0.0
        
        return {
            'total_remnants': total_remnants,
            'available_remnants': available_remnants,
            'used_remnants': used_remnants,
            'total_area': total_area,
            'available_area': available_area,
            'used_area': used_area,
            'avg_area': avg_area,
            'utilization_rate': (used_area / total_area * 100.0) if total_area > 0 else 0.0
        }
    
    def export_remnants_report(self, filename: str = "remnants_report.json"):
        """Экспорт отчета об остатках"""
        import json
        
        report = {
            'statistics': self.get_remnants_statistics(),
            'remnants': [
                {
                    'id': r.remnant_id,
                    'sheet_id': r.sheet_id,
                    'area': r.area,
                    'bounding_box': r.bounding_box,
                    'material_type': r.material_type,
                    'is_available': r.is_available,
                    'used_for_parts': r.used_for_parts
                }
                for r in self.remnants
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Отчет об остатках сохранен в {filename}")
    
    def __len__(self):
        return len(self.remnants)
    
    def __str__(self):
        stats = self.get_remnants_statistics()
        return (f"RemnantManager(remnants={len(self)}, "
                f"available={stats['available_remnants']}, "
                f"utilization={stats['utilization_rate']:.1f}%)")
