import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Set, Any
from core.geometry import PolygonShape
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import json
import re

class ConstraintManager:
    """
    Менеджер технологических ограничений для задач раскроя-упаковки
    
    Обеспечивает централизованное управление всеми ограничениями:
    - минимальные зазоры
    - ориентационные барьеры
    - приоритеты размещения
    - дефектные зоны
    - антисимметричные фигуры (анизотропные материалы)
    """
    
    def __init__(self, sheet_size: Tuple[float, float]):
        """
        Инициализация менеджера ограничений
        
        :param sheet_size: Размеры листа (ширина, высота)
        """
        self.sheet_size = sheet_size
        
        # Минимальные зазоры (зависит от технологии резки)
        self.min_gaps = {
            'laser': 0.2,      # мм
            'plasma': 2.0,     # мм
            'waterjet': 1.0,   # мм
            'default': 1.0     # мм
        }
        self.current_technology = 'default'
        self.current_gap = self.min_gaps[self.current_technology]
        
        # Ограничения на ориентацию
        self.orientation_constraints = {}  # {shape_id: List[float]}
        self.anisotropic_materials = set()  # Множество ID анизотропных материалов
        
        # Приоритеты размещения
        self.placement_priorities = {}  # {shape_id: int}
        
        # Дефектные зоны
        self.defect_zones = []  # Список полигонов дефектных зон
        
        # Запрещенные области (технологические требования)
        self.forbidden_areas = []  # Список полигонов запрещенных областей
        
        # Стандарты и требования
        self.standards = {
            'GOST_R_56079-2014': True,  # Требования к раскрою листовых материалов
            'GOST_R_56081-2014': True,  # Требования к минимизации отходов
            'ISO_10303-21': True,       # STEP формат
            'kerf_compensation': True   # Компенсация ширины реза
        }
    
    def set_technology(self, technology: str):
        """
        Установка технологии резки и соответствующего минимального зазора
        
        :param technology: 'laser', 'plasma', 'waterjet' или 'default'
        """
        if technology in self.min_gaps:
            self.current_technology = technology
            self.current_gap = self.min_gaps[technology]
            print(f"Установлена технология резки: {technology}, минимальный зазор: {self.current_gap} мм")
        else:
            raise ValueError(f"Неизвестная технология резки: {technology}. Доступные: {list(self.min_gaps.keys())}")
    
    def set_custom_gap(self, gap: float):
        """
        Установка пользовательского минимального зазора
        
        :param gap: Значение зазора в мм
        """
        if gap < 0:
            raise ValueError("Минимальный зазор не может быть отрицательным")
        self.current_gap = gap
        print(f"Установлен пользовательский минимальный зазор: {gap} мм")
    
    def add_orientation_constraint(self, shape_id: str, allowed_angles: List[float]):
        """
        Добавление ограничения на ориентацию фигуры
        
        :param shape_id: Идентификатор фигуры
        :param allowed_angles: Список допустимых углов в градусах
        """
        # Нормализация углов к диапазону [0, 360)
        normalized_angles = [angle % 360 for angle in allowed_angles]
        # Удаление дубликатов и сортировка
        self.orientation_constraints[shape_id] = sorted(set(normalized_angles))
        print(f"Добавлено ограничение ориентации для {shape_id}: {normalized_angles}°")
    
    def mark_as_anisotropic(self, shape_id: str):
        """
        Пометка фигуры как анизотропного материала
        (например, композиты, прокат)
        
        :param shape_id: Идентификатор фигуры
        """
        self.anisotropic_materials.add(shape_id)
        print(f"Фигура {shape_id} помечена как анизотропный материал")
    
    def set_placement_priority(self, shape_id: str, priority: int):
        """
        Установка приоритета размещения фигуры
        
        :param shape_id: Идентификатор фигуры
        :param priority: Целое число (чем выше, тем выше приоритет)
        """
        if priority < 0:
            raise ValueError("Приоритет не может быть отрицательным")
        self.placement_priorities[shape_id] = priority
        print(f"Установлен приоритет {priority} для фигуры {shape_id}")
    
    def add_defect_zone(self, contour: List[Tuple[float, float]], defect_type: str = "general"):
        """
        Добавление дефектной зоны на лист
        
        :param contour: Контур дефектной зоны в формате [(x1,y1), (x2,y2), ...]
        :param defect_type: Тип дефекта: 'crack', 'hole', 'inclusion', 'general'
        """
        polygon = Polygon(contour)
        if not polygon.is_valid:
            raise ValueError("Невалидный контур дефектной зоны")
        
        defect_zone = {
            'polygon': polygon,
            'type': defect_type,
            'area': polygon.area
        }
        self.defect_zones.append(defect_zone)
        print(f"Добавлена дефектная зона типа '{defect_type}' площадью {polygon.area:.2f} мм²")
    
    def add_forbidden_area(self, contour: List[Tuple[float, float]], reason: str = "technological"):
        """
        Добавление запрещенной области на лист
        
        :param contour: Контур запрещенной области в формате [(x1,y1), (x2,y2), ...]
        :param reason: Причина запрета: 'edge_distance', 'clamping', 'technological'
        """
        polygon = Polygon(contour)
        if not polygon.is_valid:
            raise ValueError("Невалидный контур запрещенной области")
        
        forbidden_area = {
            'polygon': polygon,
            'reason': reason
        }
        self.forbidden_areas.append(forbidden_area)
        print(f"Добавлена запрещенная область по причине '{reason}'")
    
    def check_clearance(self, shape1: PolygonShape, shape2: PolygonShape) -> bool:
        """
        Проверка соблюдения минимального зазора между двумя фигурами
        
        :param shape1, shape2: Фигуры для проверки
        :return: True если зазор соблюдается
        """
        distance = shape1.polygon.distance(shape2.polygon)
        return distance >= self.current_gap
    
    def check_defect_clearance(self, shape: PolygonShape) -> bool:
        """
        Проверка соблюдения зазора до всех дефектных зон
        
        :param shape: Фигура для проверки
        :return: True если фигура не пересекает дефектные зоны
        """
        for defect in self.defect_zones:
            distance = shape.polygon.distance(defect['polygon'])
            if distance < self.current_gap * 1.5:  # Усиленный зазор для дефектов
                return False
        return True
    
    def check_forbidden_areas(self, shape: PolygonShape) -> bool:
        """
        Проверка пересечения с запрещенными областями
        
        :param shape: Фигура для проверки
        :return: True если фигура не пересекает запрещенные области
        """
        for area in self.forbidden_areas:
            if shape.polygon.intersects(area['polygon']):
                return False
        return True
    
    def find_closest_allowed_angle(self, shape_id: str, current_angle: float) -> float:
        """
        Нахождение ближайшего допустимого угла для фигуры с ограничениями на ориентацию
        
        :param shape_id: Идентификатор фигуры
        :param current_angle: Текущий угол в градусах
        :return: Ближайший допустимый угол в градусах
        """
        if shape_id not in self.orientation_constraints:
            return current_angle
        
        allowed_angles = self.orientation_constraints[shape_id]
        current_angle = current_angle % 360
        
        # Найти ближайший допустимый угол
        min_diff = float('inf')
        closest_angle = allowed_angles[0]
        
        for angle in allowed_angles:
            diff = min(abs(angle - current_angle), 360 - abs(angle - current_angle))
            if diff < min_diff:
                min_diff = diff
                closest_angle = angle
        
        return closest_angle
    
    def get_shape_priority(self, shape_id: str) -> int:
        """
        Получение приоритета фигуры (по умолчанию 1)
        
        :param shape_id: Идентификатор фигуры
        :return: Приоритет размещения
        """
        return self.placement_priorities.get(shape_id, 1)
    
    def get_sheet_boundaries(self) -> Tuple[float, float, float, float]:
        """
        Получение границ листа с учетом минимального отступа от краев
        
        :return: (min_x, min_y, max_x, max_y)
        """
        # Минимальный отступ от края листа (обычно равен минимальному зазору)
        edge_margin = self.current_gap
        
        return (
            edge_margin,
            edge_margin,
            self.sheet_size[0] - edge_margin,
            self.sheet_size[1] - edge_margin
        )
    
    def validate_placement(self, shape: PolygonShape, position: np.ndarray, angle: float) -> bool:
        """
        Комплексная проверка допустимости размещения фигуры
        
        :param shape: Фигура
        :param position: Позиция центра масс (x, y)
        :param angle: Угол поворота в градусах
        :return: True если размещение допустимо
        """
        # Применение трансформации к фигуре
        transformed_shape = shape.apply_transformation(position, angle)
        
        # 1. Проверка границ листа
        min_x, min_y, max_x, max_y = self.get_sheet_boundaries()
        shape_min_x, shape_min_y, shape_max_x, shape_max_y = transformed_shape.get_bounding_box()
        
        if shape_min_x < min_x or shape_min_y < min_y or shape_max_x > max_x or shape_max_y > max_y:
            return False
        
        # 2. Проверка пересечения с дефектными зонами
        if not self.check_defect_clearance(transformed_shape):
            return False
        
        # 3. Проверка пересечения с запрещенными областями
        if not self.check_forbidden_areas(transformed_shape):
            return False
        
        return True
    
    def apply_constraints_to_force(self, force: np.ndarray, torque: float, shape_id: str) -> Tuple[np.ndarray, float]:
        """
        Применение технологических ограничений к силе и моменту
        
        :param force: Исходный вектор силы
        :param torque: Исходный момент
        :param shape_id: Идентификатор фигуры
        :return: (скорректированный вектор силы, скорректированный момент)
        """
        # Корректировка момента для анизотропных материалов
        if shape_id in self.anisotropic_materials and shape_id in self.orientation_constraints:
            allowed_angles = self.orientation_constraints[shape_id]
            # Усиление момента для принудительного выравнивания по допустимым углам
            torque *= 2.0
        
        return force, torque
    
    def export_to_json(self, filename: str):
        """
        Экспорт ограничений в JSON файл
        
        :param filename: Имя файла для экспорта
        """
        export_data = {
            'sheet_size': self.sheet_size,
            'technology': self.current_technology,
            'min_gap': self.current_gap,
            'orientation_constraints': {k: v for k, v in self.orientation_constraints.items()},
            'anisotropic_materials': list(self.anisotropic_materials),
            'placement_priorities': {k: v for k, v in self.placement_priorities.items()},
            'defect_zones': [
                {
                    'contour': list(zone['polygon'].exterior.coords)[:-1],  # Убираем последнюю точку (дублирует первую)
                    'type': zone['type']
                } for zone in self.defect_zones
            ],
            'forbidden_areas': [
                {
                    'contour': list(area['polygon'].exterior.coords)[:-1],
                    'reason': area['reason']
                } for area in self.forbidden_areas
            ],
            'standards': self.standards
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Технологические ограничения экспортированы в {filename}")
    
    def import_from_json(self, filename: str):
        """
        Импорт ограничений из JSON файла
        
        :param filename: Имя файла для импорта
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Очистка текущих ограничений
            self.orientation_constraints = {}
            self.anisotropic_materials = set()
            self.placement_priorities = {}
            self.defect_zones = []
            self.forbidden_areas = []
            
            # Импорт данных
            self.sheet_size = tuple(data.get('sheet_size', self.sheet_size))
            technology = data.get('technology', 'default')
            if technology in self.min_gaps:
                self.current_technology = technology
                self.current_gap = data.get('min_gap', self.min_gaps[technology])
            
            # Импорт ограничений ориентации
            for shape_id, angles in data.get('orientation_constraints', {}).items():
                self.add_orientation_constraint(shape_id, angles)
            
            # Импорт анизотропных материалов
            for shape_id in data.get('anisotropic_materials', []):
                self.mark_as_anisotropic(shape_id)
            
            # Импорт приоритетов
            for shape_id, priority in data.get('placement_priorities', {}).items():
                self.set_placement_priority(shape_id, priority)
            
            # Импорт дефектных зон
            for zone in data.get('defect_zones', []):
                self.add_defect_zone(zone['contour'], zone['type'])
            
            # Импорт запрещенных областей
            for area in data.get('forbidden_areas', []):
                self.add_forbidden_area(area['contour'], area['reason'])
            
            # Импорт стандартов
            standards = data.get('standards', {})
            for std, enabled in standards.items():
                if std in self.standards:
                    self.standards[std] = enabled
            
            print(f"Технологические ограничения импортированы из {filename}")
            print(f"Загружено: {len(self.orientation_constraints)} ограничений ориентации, "
                  f"{len(self.anisotropic_materials)} анизотропных материалов, "
                  f"{len(self.defect_zones)} дефектных зон")
        
        except Exception as e:
            print(f"Ошибка при импорте ограничений из {filename}: {e}")
            raise
    
    def import_from_dxf(self, filename: str):
        """
        Импорт технологических ограничений из DXF файла
        
        :param filename: Имя DXF файла для импорта
        """
        try:
            import ezdxf
            
            doc = ezdxf.readfile(filename)
            msp = doc.modelspace()
            
            print(f"Импорт ограничений из DXF: {filename}")
            
            # Поиск слоев с ограничениями
            constraint_layers = {
                'DEFECTS': 'defect_zones',
                'FORBIDDEN': 'forbidden_areas',
                'CONSTRAINTS': 'constraints'
            }
            
            # Импорт дефектных зон
            if 'DEFECTS' in doc.layers:
                for entity in msp.query('LWPOLYLINE CIRCLE ELLIPSE').filter(lambda e: e.dxf.layer == 'DEFECTS'):
                    try:
                        if entity.dxftype() == 'LWPOLYLINE':
                            contour = [(point[0], point[1]) for point in entity.get_points()]
                            self.add_defect_zone(contour, "imported")
                        elif entity.dxftype() == 'CIRCLE':
                            center = (entity.dxf.center.x, entity.dxf.center.y)
                            radius = entity.dxf.radius
                            # Аппроксимация круга полигоном
                            contour = [(center[0] + radius * np.cos(2*np.pi*i/32), 
                                       center[1] + radius * np.sin(2*np.pi*i/32)) for i in range(33)]
                            self.add_defect_zone(contour, "imported")
                        elif entity.dxftype() == 'ELLIPSE':
                            # Простая аппроксимация эллипса
                            center = (entity.dxf.center.x, entity.dxf.center.y)
                            major_axis = entity.dxf.major_axis
                            ratio = entity.dxf.ratio
                            # Создание контура эллипса
                            contour = []
                            for i in range(33):
                                angle = 2 * np.pi * i / 32
                                x = center[0] + major_axis[0] * np.cos(angle)
                                y = center[1] + major_axis[1] * ratio * np.sin(angle)
                                contour.append((x, y))
                            self.add_defect_zone(contour, "imported")
                    except Exception as e:
                        print(f"Ошибка при импорте дефектной зоны: {e}")
            
            # Импорт запрещенных областей
            if 'FORBIDDEN' in doc.layers:
                for entity in msp.query('LWPOLYLINE').filter(lambda e: e.dxf.layer == 'FORBIDDEN'):
                    try:
                        contour = [(point[0], point[1]) for point in entity.get_points()]
                        self.add_forbidden_area(contour, "imported")
                    except Exception as e:
                        print(f"Ошибка при импорте запрещенной области: {e}")
            
            # Импорт текстовых ограничений (приоритеты, ориентация)
            if 'CONSTRAINTS' in doc.layers:
                for entity in msp.query('TEXT').filter(lambda e: e.dxf.layer == 'CONSTRAINTS'):
                    try:
                        text = entity.dxf.text
                        # Поиск шаблонов: "SHAPE_ID:priority=2", "SHAPE_ID:angles=0,90"
                        if ':' in text:
                            shape_id_part, constraint_part = text.split(':', 1)
                            shape_id = shape_id_part.strip()
                            
                            if 'priority=' in constraint_part:
                                priority = int(re.search(r'priority=(\d+)', constraint_part).group(1))
                                self.set_placement_priority(shape_id, priority)
                            
                            if 'angles=' in constraint_part:
                                angles_str = re.search(r'angles=([0-9,\s]+)', constraint_part).group(1)
                                angles = [float(a.strip()) for a in angles_str.split(',') if a.strip()]
                                self.add_orientation_constraint(shape_id, angles)
                            
                            if 'anisotropic' in constraint_part.lower():
                                self.mark_as_anisotropic(shape_id)
                    except Exception as e:
                        print(f"Ошибка при импорте текстовых ограничений: {e}")
            
            print(f"Завершен импорт из DXF: {len(self.defect_zones)} дефектных зон, "
                  f"{len(self.forbidden_areas)} запрещенных областей")
        
        except ImportError:
            print("Ошибка: Не установлен модуль ezdxf. Установите: pip install ezdxf")
            raise
        except Exception as e:
            print(f"Ошибка при импорте из DXF: {e}")
            raise
    
    def generate_constraints_report(self) -> Dict[str, Any]:
        """
        Генерация отчета по технологическим ограничениям
        
        :return: Словарь с информацией об ограничениях
        """
        report = {
            'technology': self.current_technology,
            'min_gap': self.current_gap,
            'total_defect_zones': len(self.defect_zones),
            'total_defect_area': sum(zone['area'] for zone in self.defect_zones),
            'total_forbidden_areas': len(self.forbidden_areas),
            'constrained_shapes': len(self.orientation_constraints),
            'anisotropic_shapes': len(self.anisotropic_materials),
            'priority_levels': len(set(self.placement_priorities.values())),
            'standards_compliance': self.standards
        }
        
        # Детализация по дефектам
        defect_types = {}
        for zone in self.defect_zones:
            defect_types[zone['type']] = defect_types.get(zone['type'], 0) + 1
        report['defect_types'] = defect_types
        
        # Детализация по ограничениям ориентации
        angle_distributions = {}
        for shape_id, angles in self.orientation_constraints.items():
            num_angles = len(angles)
            angle_distributions[num_angles] = angle_distributions.get(num_angles, 0) + 1
        report['angle_distributions'] = angle_distributions
        
        return report
    
    def __str__(self):
        """Строковое представление для отладки"""
        report = self.generate_constraints_report()
        return (f"ConstraintManager:\n"
                f"  Технология резки: {report['technology']} (зазор: {report['min_gap']} мм)\n"
                f"  Дефектные зоны: {report['total_defect_zones']} (площадь: {report['total_defect_area']:.2f} мм²)\n"
                f"  Ограничения ориентации: {report['constrained_shapes']} фигур\n"
                f"  Анизотропные материалы: {report['anisotropic_shapes']} фигур\n"
                f"  Приоритеты размещения: {report['priority_levels']} уровней\n"
                f"  Запрещенные области: {report['total_forbidden_areas']}")