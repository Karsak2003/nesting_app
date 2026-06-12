
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from core.geometry import PolygonShape
from core.constraints import ConstraintManager
import logging
import uuid

"""
Модуль для управления партией листовых материалов
Согласно разделу 2.5.5 и 3.3.9, учет характеристик каждого листа критичен для промышленного применения
"""


logger = logging.getLogger(__name__)

class Sheet:
    """
    Класс для представления отдельного листа материала
    
    Согласно разделу 2.5.5, листы могут иметь различные характеристики:
    - Размеры (ширина, высота)
    - Дефектные зоны
    - Тип материала
    - Технология резки
    - Приоритет использования
    - Стоимость
    """
    
    def __init__(self, 
                 sheet_id: str,
                 width: float, 
                 height: float,
                 material_type: str = "steel",
                 cutting_technology: str = "laser",
                 cost: float = 0.0,
                 priority: int = 1,
                 defects: Optional[List[PolygonShape]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация листа материала
        
        :param sheet_id: Уникальный идентификатор листа
        :param width: Ширина листа в мм
        :param height: Высота листа в мм
        :param material_type: Тип материала
        :param cutting_technology: Технология резки
        :param cost: Стоимость листа
        :param priority: Приоритет использования (чем выше, тем раньше используется)
        :param defects: Список дефектных зон на листе
        :param metadata: Дополнительные метаданные
        """
        self.sheet_id = sheet_id
        self.width = width
        self.height = height
        self.material_type = material_type
        self.cutting_technology = cutting_technology
        self.cost = cost
        self.priority = priority
        self.defects = defects if defects else []
        self.metadata = metadata if metadata else {}
        
        # Статус листа
        self.is_used = False
        self.utilization = 0.0
        self.placed_parts = []
        self.remaining_area = width * height
        
        # Расчет минимального зазора в зависимости от технологии
        self.min_gap = self._get_min_gap_for_technology(cutting_technology)
    
    def _get_min_gap_for_technology(self, technology: str) -> float:
        """Получение минимального зазора для технологии резки"""
        gaps = {
            'laser': 0.2,
            'plasma': 2.0,
            'waterjet': 1.0,
            'default': 1.0
        }
        return gaps.get(technology, 1.0)
    
    def get_sheet_size(self) -> Tuple[float, float]:
        """Получение размеров листа"""
        return (self.width, self.height)
    
    def add_defect(self, defect: PolygonShape):
        """Добавление дефектной зоны на лист"""
        self.defects.append(defect)
        logger.debug(f"Добавлена дефектная зона на лист {self.sheet_id}")
    
    def mark_as_used(self, utilization: float, placed_parts: List[Any]):
        """Пометка листа как использованного"""
        self.is_used = True
        self.utilization = utilization
        self.placed_parts = placed_parts
        self.remaining_area = self.width * self.height * (1 - utilization / 100.0)
        logger.info(f"Лист {self.sheet_id} использован: утилизация {utilization:.2f}%")
    
    def get_remaining_rectangles(self) -> List[Tuple[float, float, float, float]]:
        """
        Получение списка оставшихся прямоугольных областей на листе
        
        Возвращает список прямоугольников в формате (x, y, width, height)
        """
        # Упрощенная реализация - возвращает один прямоугольник, занимаемый оставшейся площадью
        # В реальной реализации потребуется анализ свободного пространства
        remaining_width = self.width * np.sqrt(self.remaining_area / (self.width * self.height))
        remaining_height = self.remaining_area / remaining_width
        
        return [(0, 0, remaining_width, remaining_height)]
    
    def __str__(self):
        return (f"Sheet(id={self.sheet_id}, size={self.width}x{self.height}mm, "
                f"material={self.material_type}, utilization={self.utilization:.1f}%, "
                f"priority={self.priority})")

class SheetBatchManager:
    """
    Менеджер партии листовых материалов
    
    Управляет набором листов, их характеристиками и распределением деталей по листам.
    Согласно разделу 3.3.9, учет характеристик каждого листа критичен для промышленного применения.
    """
    
    def __init__(self, sheets: Optional[List[Sheet]] = None):
        """
        Инициализация менеджера партии листов
        
        :param sheets: Список листов в партии
        """
        self.sheets = sheets if sheets else []
        self.sheet_index = {}  # Индекс листов по ID
        self._build_index()
    
    def _build_index(self):
        """Построение индекса листов по ID"""
        self.sheet_index = {sheet.sheet_id: sheet for sheet in self.sheets}
    
    def add_sheet(self, sheet: Sheet):
        """Добавление листа в партию"""
        self.sheets.append(sheet)
        self.sheet_index[sheet.sheet_id] = sheet
        logger.info(f"Добавлен лист {sheet.sheet_id} в партию")
    
    def add_sheets_from_config(self, config_file: str):
        """
        Добавление листов из конфигурационного файла
        
        Формат конфигурации:
        {
            "sheets": [
                {
                    "id": "sheet_001",
                    "width": 2000,
                    "height": 1000,
                    "material_type": "steel",
                    "cutting_technology": "laser",
                    "cost": 100000,
                    "priority": 1,
                    "defects": [
                        {"contour": [[x1,y1], [x2,y2], ...], "type": "crack"}
                    ]
                },
                ...
            ]
        }
        """
        import json
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for sheet_config in config.get('sheets', []):
                sheet_id = sheet_config.get('id', str(uuid.uuid4()))
                width = sheet_config.get('width', 2000.0)
                height = sheet_config.get('height', 1000.0)
                material_type = sheet_config.get('material_type', 'steel')
                cutting_technology = sheet_config.get('cutting_technology', 'laser')
                cost = sheet_config.get('cost', 0.0)
                priority = sheet_config.get('priority', 1)
                
                # Создание листа
                sheet = Sheet(
                    sheet_id=sheet_id,
                    width=width,
                    height=height,
                    material_type=material_type,
                    cutting_technology=cutting_technology,
                    cost=cost,
                    priority=priority
                )
                
                # Добавление дефектных зон
                for defect_config in sheet_config.get('defects', []):
                    contour = defect_config.get('contour', [])
                    if contour:
                        defect_shape = PolygonShape(contour, name=f"defect_{sheet_id}")
                        sheet.add_defect(defect_shape)
                
                self.add_sheet(sheet)
            
            logger.info(f"Загружено {len(config.get('sheets', []))} листов из конфигурации")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации листов: {e}")
            raise
    
    def get_available_sheets(self, material_type: Optional[str] = None) -> List[Sheet]:
        """
        Получение списка доступных (неиспользованных) листов
        
        :param material_type: Тип материала для фильтрации
        :return: Список доступных листов
        """
        available = [sheet for sheet in self.sheets if not sheet.is_used]
        
        if material_type:
            available = [sheet for sheet in available if sheet.material_type == material_type]
        
        # Сортировка по приоритету и стоимости
        available.sort(key=lambda s: (s.priority, s.cost))
        
        return available
    
    def get_sheet_by_id(self, sheet_id: str) -> Optional[Sheet]:
        """Получение листа по идентификатору"""
        return self.sheet_index.get(sheet_id)
    
    def mark_sheet_as_used(self, sheet_id: str, utilization: float, placed_parts: List[Any]):
        """Пометка листа как использованного"""
        sheet = self.get_sheet_by_id(sheet_id)
        if sheet:
            sheet.mark_as_used(utilization, placed_parts)
            logger.info(f"Лист {sheet_id} помечен как использованный")
        else:
            logger.warning(f"Лист {sheet_id} не найден в партии")
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по партии листов
        
        :return: Словарь со статистикой
        """
        total_sheets = len(self.sheets)
        used_sheets = sum(1 for sheet in self.sheets if sheet.is_used)
        unused_sheets = total_sheets - used_sheets
        
        total_area = sum(sheet.width * sheet.height for sheet in self.sheets)
        used_area = sum(sheet.width * sheet.height * sheet.utilization / 100.0 
                       for sheet in self.sheets if sheet.is_used)
        
        total_cost = sum(sheet.cost for sheet in self.sheets if sheet.is_used)
        
        utilization_by_sheet = [sheet.utilization for sheet in self.sheets if sheet.is_used]
        avg_utilization = np.mean(utilization_by_sheet) if utilization_by_sheet else 0.0
        min_utilization = np.min(utilization_by_sheet) if utilization_by_sheet else 0.0
        max_utilization = np.max(utilization_by_sheet) if utilization_by_sheet else 0.0
        
        return {
            'total_sheets': total_sheets,
            'used_sheets': used_sheets,
            'unused_sheets': unused_sheets,
            'total_area': total_area,
            'used_area': used_area,
            'utilization_rate': (used_area / total_area * 100.0) if total_area > 0 else 0.0,
            'total_cost': total_cost,
            'avg_utilization': avg_utilization,
            'min_utilization': min_utilization,
            'max_utilization': max_utilization,
            'sheets_by_material': self._group_sheets_by_material(),
            'sheets_by_technology': self._group_sheets_by_technology()
        }
    
    def _group_sheets_by_material(self) -> Dict[str, int]:
        """Группировка листов по типу материала"""
        groups = {}
        for sheet in self.sheets:
            groups[sheet.material_type] = groups.get(sheet.material_type, 0) + 1
        return groups
    
    def _group_sheets_by_technology(self) -> Dict[str, int]:
        """Группировка листов по технологии резки"""
        groups = {}
        for sheet in self.sheets:
            groups[sheet.cutting_technology] = groups.get(sheet.cutting_technology, 0) + 1
        return groups
    
    def export_batch_report(self, filename: str = "sheet_batch_report.json"):
        """Экспорт отчета о партии листов"""
        import json
        
        report = {
            'batch_statistics': self.get_batch_statistics(),
            'sheets': [
                {
                    'id': sheet.sheet_id,
                    'size': [sheet.width, sheet.height],
                    'material_type': sheet.material_type,
                    'cutting_technology': sheet.cutting_technology,
                    'cost': sheet.cost,
                    'priority': sheet.priority,
                    'is_used': sheet.is_used,
                    'utilization': sheet.utilization,
                    'num_defects': len(sheet.defects),
                    'num_placed_parts': len(sheet.placed_parts)
                }
                for sheet in self.sheets
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Отчет о партии листов сохранен в {filename}")
    
    def __len__(self):
        return len(self.sheets)
    
    def __str__(self):
        stats = self.get_batch_statistics()
        return (f"SheetBatchManager(sheets={len(self.sheets)}, "
                f"used={stats['used_sheets']}, utilization={stats['utilization_rate']:.1f}%)")

