
"""
Модуль для верификации решений раскроя-упаковки
Согласно разделу 2.2.9, верификация критична для промышленного внедрения
"""

import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Dict, Tuple, Optional, Any
import logging
from core.geometry import PolygonShape
from core.constraints import ConstraintManager
from utils.nfp import build_nfp_for_pair

logger = logging.getLogger(__name__)

class SolutionVerifier:
    """
    Класс для верификации допустимости решения раскроя-упаковки
    
    Проверяет:
    - Отсутствие коллизий между фигурами
    - Соблюдение минимальных зазоров
    - Соблюдение ориентационных ограничений
    - Отсутствие пересечения с дефектными зонами
    - Укладку в границы листа
    """
    
    def __init__(self, constraint_manager: ConstraintManager):
        """
        Инициализация верификатора
        
        :param constraint_manager: Менеджер технологических ограничений
        """
        self.constraint_manager = constraint_manager
        self.verification_results = {}
        self.errors = []
        self.warnings = []
    
    def verify_solution(self, placements: List[Dict[str, Any]], 
                       sheet_size: Tuple[float, float],
                       defect_zones: Optional[List[PolygonShape]] = None) -> bool:
        """
        Полная верификация решения раскроя-упаковки
        
        :param placements: Список размещенных фигур с позициями и углами
        :param sheet_size: Размеры листа (ширина, высота)
        :param defect_zones: Список дефектных зон
        :return: True если решение допустимо
        """
        logger.info("Начало верификации решения...")
        
        # Сброс результатов предыдущей верификации
        self.verification_results = {}
        self.errors = []
        self.warnings = []
        
        # 1. Проверка укладки в границы листа
        self._verify_sheet_boundaries(placements, sheet_size)
        
        # 2. Проверка коллизий между фигурами
        self._verify_no_collisions(placements)
        
        # 3. Проверка минимальных зазоров
        self._verify_min_gaps(placements)
        
        # 4. Проверка ориентационных ограничений
        self._verify_orientation_constraints(placements)
        
        # 5. Проверка дефектных зон
        if defect_zones:
            self._verify_defect_zones(placements, defect_zones)
        
        # 6. Проверка приоритетов (если применимо)
        self._verify_priorities(placements)
        
        # Формирование итогового результата
        is_valid = len(self.errors) == 0
        
        if is_valid:
            logger.info("✅ Решение прошло верификацию. Все ограничения соблюдены.")
        else:
            logger.error(f"❌ Решение НЕДОПУСТИМО. Обнаружено ошибок: {len(self.errors)}, предупреждений: {len(self.warnings)}")
            for error in self.errors:
                logger.error(f"  • {error}")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        return is_valid
    
    def _verify_sheet_boundaries(self, placements: List[Dict[str, Any]], 
                                sheet_size: Tuple[float, float]):
        """Проверка укладки всех фигур в границы листа"""
        width, height = sheet_size
        min_gap = self.constraint_manager.current_gap
        
        for placement in placements:
            shape = placement['shape']
            position = placement['position']
            angle = placement['angle']
            
            # Получение трансформированной фигуры
            transformed_shape = shape.apply_transformation(position, angle)
            min_x, min_y, max_x, max_y = transformed_shape.get_bounding_box()
            # Проверка границ с учетом минимального зазора
            if min_x < min_gap:
                self.errors.append(f"Фигура {placement.get('name', 'unnamed')} выходит за левую границу листа")
            if min_y < min_gap:
                self.errors.append(f"Фигура {placement.get('name', 'unnamed')} выходит за нижнюю границу листа")
            if max_x > width - min_gap:
                self.errors.append(f"Фигура {placement.get('name', 'unnamed')} выходит за правую границу листа")
            if max_y > height - min_gap:
                self.errors.append(f"Фигура {placement.get('name', 'unnamed')} выходит за верхнюю границу листа")
    
    def _verify_no_collisions(self, placements: List[Dict[str, Any]]):
        """Проверка отсутствия коллизий между фигурами"""
        n = len(placements)
        
        for i in range(n):
            for j in range(i + 1, n):
                shape_i = placements[i]['shape'].apply_transformation(
                    placements[i]['position'], placements[i]['angle']
                )
                shape_j = placements[j]['shape'].apply_transformation(
                    placements[j]['position'], placements[j]['angle']
                )
                
                # Проверка пересечения полигонов
                if shape_i.polygon.intersects(shape_j.polygon):
                    intersection_area = shape_i.polygon.intersection(shape_j.polygon).area
                    if intersection_area > 1e-6:  # Порог для учета пересечения
                        self.errors.append(
                            f"Коллизия между {placements[i].get('name', 'unnamed')} и "
                            f"{placements[j].get('name', 'unnamed')} (площадь пересечения: {intersection_area:.2f} мм²)"
                        )
    
    def _verify_min_gaps(self, placements: List[Dict[str, Any]]):
        """Проверка соблюдения минимальных зазоров между фигурами"""
        min_gap = self.constraint_manager.current_gap
        
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                shape_i = placements[i]['shape'].apply_transformation(
                    placements[i]['position'], placements[i]['angle']
                )
                shape_j = placements[j]['shape'].apply_transformation(
                    placements[j]['position'], placements[j]['angle']
                )
                
                # Расчет минимального расстояния между фигурами
                distance = shape_i.polygon.distance(shape_j.polygon)
                
                if distance < min_gap - 1e-6:  # С учетом погрешности
                    gap_violation = min_gap - distance
                    self.warnings.append(
                        f"Нарушение минимального зазора между {placements[i].get('name', 'unnamed')} и "
                        f"{placements[j].get('name', 'unnamed')}: {gap_violation:.2f} мм"
                    )
    
    def _verify_orientation_constraints(self, placements: List[Dict[str, Any]]):
        """Проверка соблюдения ориентационных ограничений"""
        for placement in placements:
            shape_id = placement.get('name', str(id(placement)))
            angle = placement['angle'] % 360
            
            # Получение допустимых углов для фигуры
            allowed_angles = self.constraint_manager.orientation_constraints.get(shape_id)
            
            if allowed_angles:
                # Поиск ближайшего допустимого угла
                closest_angle = min(allowed_angles, key=lambda x: min(abs(x - angle), 360 - abs(x - angle)))
                angle_diff = min(abs(angle - closest_angle), 360 - abs(angle - closest_angle))
                
                if angle_diff > 1e-6:  # С учетом погрешности
                    self.errors.append(
                        f"Нарушение ориентационного ограничения для {shape_id}: "
                        f"фактический угол {angle:.2f}°, ближайший допустимый {closest_angle:.2f}°"
                    )
    
    def _verify_defect_zones(self, placements: List[Dict[str, Any]], 
                            defect_zones: List[PolygonShape]):
        """Проверка отсутствия пересечения с дефектными зонами"""
        min_gap = self.constraint_manager.current_gap * 1.5  # Усиленный зазор для дефектов
        
        for placement in placements:
            transformed_shape = placement['shape'].apply_transformation(
                placement['position'], placement['angle']
            )
            
            for defect in defect_zones:
                # Проверка пересечения с дефектной зоной
                if transformed_shape.polygon.intersects(defect.polygon):
                    self.errors.append(
                        f"Фигура {placement.get('name', 'unnamed')} пересекает дефектную зону"
                    )
                else:
                    # Проверка минимального зазора до дефектной зоны
                    distance = transformed_shape.polygon.distance(defect.polygon)
                    if distance < min_gap:
                        self.warnings.append(
                            f"Фигура {placement.get('name', 'unnamed')} слишком близко к дефектной зоне "
                            f"(расстояние: {distance:.2f} мм, требуется: {min_gap:.2f} мм)"
                        )
    
    def _verify_priorities(self, placements: List[Dict[str, Any]]):
        """Проверка соблюдения приоритетов размещения (если применимо)"""
        # Эта проверка зависит от конкретной стратегии размещения
        # В данном случае просто проверяем, что приоритеты заданы корректно
        for placement in placements:
            priority = placement.get('priority', 1)
            if priority < 1:
                self.errors.append(f"Некорректный приоритет для {placement.get('name', 'unnamed')}: {priority}")
    
    def get_verification_report(self) -> Dict[str, Any]:
        """Получение подробного отчета о верификации"""
        return {
            'is_valid': len(self.errors) == 0,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.verification_results
        }
    
    def export_verification_report(self, filename: str = "verification_report.txt"):
        """Экспорт отчета о верификации в файл"""
        report = self.get_verification_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== ОТЧЕТ О ВЕРИФИКАЦИИ РЕШЕНИЯ РАСКРОЯ ===\n\n")
            f.write(f"Статус: {'ДОПУСТИМО' if report['is_valid'] else 'НЕДОПУСТИМО'}\n")
            f.write(f"Ошибок: {report['errors_count']}\n")
            f.write(f"Предупреждений: {report['warnings_count']}\n\n")
            
            if report['errors']:
                f.write("ОШИБКИ:\n")
                for i, error in enumerate(report['errors'], 1):
                    f.write(f"{i}. {error}\n")
                f.write("\n")
            
            if report['warnings']:
                f.write("ПРЕДУПРЕЖДЕНИЯ:\n")
                for i, warning in enumerate(report['warnings'], 1):
                    f.write(f"{i}. {warning}\n")
                f.write("\n")
            
            f.write("Верификация завершена.\n")
        
        logger.info(f"Отчет о верификации сохранен в {filename}")

def verify_solution(placements: List[Dict[str, Any]], 
                   sheet_size: Tuple[float, float],
                   constraint_manager: ConstraintManager,
                   defect_zones: Optional[List[PolygonShape]] = None) -> bool:
    """
    Функция-обертка для верификации решения
    
    :param placements: Список размещенных фигур
    :param sheet_size: Размеры листа
    :param constraint_manager: Менеджер ограничений
    :param defect_zones: Список дефектных зон
    :return: True если решение допустимо
    """
    verifier = SolutionVerifier(constraint_manager)
    return verifier.verify_solution(placements, sheet_size, defect_zones)

