
"""
Модуль для работы с составными фигурами (многосвязные области и составные геометрии)
Согласно разделу 2.1.8, поддержка составных геометрий критична для промышленного внедрения
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CompositeShape:
    """
    Класс для представления составных фигур
    
    Составная фигура может состоять из нескольких связных компонент,
    каждая из которых может иметь собственные отверстия.
    """
    
    def __init__(self, components: List[Polygon], name: str = ""):
        """
        Инициализация составной фигуры
        
        :param components: Список связных компонент (полигонов)
        :param name: Имя фигуры
        """
        self.components = components
        self.name = name
        self._validate_components()
        self._calculate_properties()
    
    def _validate_components(self):
        """Проверка корректности компонент"""
        if not self.components:
            raise ValueError("Составная фигура должна содержать хотя бы одну компоненту")
        
        for i, component in enumerate(self.components):
            if not isinstance(component, Polygon):
                raise TypeError(f"Компонента {i} должна быть экземпляром Polygon")
            if not component.is_valid:
                raise ValueError(f"Компонента {i} имеет невалидную геометрию")
    
    def _calculate_properties(self):
        """Расчет геометрических свойств составной фигуры"""
        # Общая площадь - сумма площадей компонент
        self.area = sum(component.area for component in self.components)
        
        # Центр масс - взвешенное среднее центров масс компонент
        weighted_centroid = np.array([0.0, 0.0])
        for component in self.components:
            centroid = np.array(component.centroid.coords[0])
            weighted_centroid += centroid * component.area
        
        self.centroid = weighted_centroid / self.area if self.area > 0 else np.array([0.0, 0.0])
        
        # Ограничивающий прямоугольник для всей составной фигуры
        all_coords = np.vstack([np.array(component.exterior.coords)[:-1] for component in self.components])
        self.min_x, self.min_y = all_coords.min(axis=0)
        self.max_x, self.max_y = all_coords.max(axis=0)
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Получение ограничивающего прямоугольника"""
        return (self.min_x, self.min_y, self.max_x, self.max_y)
    
    def translate(self, dx: float, dy: float) -> 'CompositeShape':
        """
        Перемещение составной фигуры
        
        :param dx: Смещение по оси X
        :param dy: Смещение по оси Y
        :return: Новая составная фигура с примененным смещением
        """
        translated_components = [
            Polygon([(x + dx, y + dy) for x, y in component.exterior.coords[:-1]])
            for component in self.components
        ]
        
        return CompositeShape(translated_components, self.name)
    
    def rotate(self, angle: float, origin: Optional[Tuple[float, float]] = None) -> 'CompositeShape':
        """
        Поворот составной фигуры
        
        :param angle: Угол поворота в градусах
        :param origin: Точка поворота (по умолчанию центр масс)
        :return: Новая составная фигура с примененным поворотом
        """
        if origin is None:
            origin = tuple(self.centroid)
        
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        rotated_components = []
        for component in self.components:
            rotated_coords = []
            for x, y in component.exterior.coords[:-1]:
                # Смещение относительно точки поворота
                x_rel = x - origin[0]
                y_rel = y - origin[1]
                
                # Поворот
                x_new = x_rel * cos_angle - y_rel * sin_angle
                y_new = x_rel * sin_angle + y_rel * cos_angle
                
                # Возврат в исходную систему координат
                rotated_coords.append((x_new + origin[0], y_new + origin[1]))
            
            rotated_components.append(Polygon(rotated_coords))
        
        return CompositeShape(rotated_components, self.name)
    
    def apply_transformation(self, position: Tuple[float, float], angle: float) -> 'CompositeShape':
        """
        Применение трансформации (перемещение и поворот)
        
        :param position: Позиция центра масс
        :param angle: Угол поворота в градусах
        :return: Трансформированная составная фигура
        """
        # Сначала поворот относительно текущего центра масс
        rotated = self.rotate(angle, tuple(self.centroid))
        
        # Затем перемещение центра масс в заданную позицию
        dx = position[0] - rotated.centroid[0]
        dy = position[1] - rotated.centroid[1]
        
        return rotated.translate(dx, dy)
    
    def to_multipolygon(self) -> MultiPolygon:
        """Преобразование в MultiPolygon для совместимости с Shapely"""
        return MultiPolygon(self.components)
    
    def __str__(self):
        return f"CompositeShape(name={self.name}, components={len(self.components)}, area={self.area:.2f})"

def create_composite_from_dxf_entity(entity, layer: str = "") -> Optional[CompositeShape]:
    """
    Создание составной фигуры из элемента DXF
    
    :param entity: Объект ezdxf entity
    :param layer: Имя слоя
    :return: Составная фигура или None
    """
    try:
        import ezdxf
        
        if entity.dxftype() == 'INSERT':
            # Обработка блока (составной фигуры)
            components = []
            
            # Извлечение геометрии из блока
            block = entity.block()
            for block_entity in block:
                if block_entity.dxftype() == 'LWPOLYLINE':
                    vertices = [(point[0], point[1]) for point in block_entity.get_points()]
                    if block_entity.is_closed:
                        components.append(Polygon(vertices))
            
            if components:
                return CompositeShape(components, name=f"dxf_composite_{layer}")
        
        return None
        
    except Exception as e:
        logger.warning(f"Ошибка при создании составной фигуры из DXF entity: {e}")
        return None

def merge_nearby_components(components: List[Polygon], 
                           distance_threshold: float = 1.0) -> List[Polygon]:
    """
    Объединение близко расположенных компонент в одну составную фигуру
    
    :param components: Список компонент
    :param distance_threshold: Порог расстояния для объединения
    :return: Список объединенных компонент
    """
    if len(components) <= 1:
        return components
    
    # Создание пространственного индекса для ускорения поиска
    from rtree import index
    
    idx = index.Index()
    for i, component in enumerate(components):
        minx, miny, maxx, maxy = component.bounds
        idx.insert(i, (minx, miny, maxx, maxy))
    
    merged = []
    used = set()
    
    for i, component in enumerate(components):
        if i in used:
            continue
        
        # Поиск близлежащих компонент
        minx, miny, maxx, maxy = component.bounds
        neighbors = list(idx.intersection(
            (minx - distance_threshold, miny - distance_threshold, 
             maxx + distance_threshold, maxy + distance_threshold)
        ))
        
        # Объединение компонент, находящихся ближе порога
        to_merge = [component]
        for j in neighbors:
            if j != i and j not in used:
                other = components[j]
                if component.distance(other) < distance_threshold:
                    to_merge.append(other)
                    used.add(j)
        
        if len(to_merge) > 1:
            # Объединение компонент
            merged_component = unary_union(to_merge)
            if isinstance(merged_component, Polygon):
                merged.append(merged_component)
            elif isinstance(merged_component, MultiPolygon):
                merged.extend(list(merged_component.geoms))
        else:
            merged.append(component)
        
        used.add(i)
    
    logger.info(f"Объединено {len(components) - len(merged)} компонент в {len(merged)} составных фигур")
    return merged