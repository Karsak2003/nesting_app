import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate

class PolygonShape:
    """
    Класс для представления геометрии фигуры с поддержкой отверстий
    """
    def __init__(self, 
                 outer_contour:list[tuple[int|float]], 
                 inner_contours:list[tuple[int|float]]=None, 
                 name:str="", 
                 priority:int = None):
        """
        Инициализация фигуры
        
        :param outer_contour: внешний контур (список точек [(x1,y1), (x2,y2), ...])
        :param inner_contours: список внутренних контуров (отверстий)
        :param name: Описание
        :param priority: Описание
        """

        self.name:str = name
        self.outer_contour:np.ndarray = np.array(outer_contour)
        self.inner_contours:list[np.ndarray] = [np.array(contour) for contour in inner_contours] if inner_contours else []
        self.priority:int = priority
        self.allowed_angles:list = []
        self.update_geometry()

        
    def update_geometry(self):
        """Обновление геометрических характеристик после изменений"""
        # Создание Shapely полигонов
        outer_polygon = Polygon(self.outer_contour)
        inner_polygons = [Polygon(contour) for contour in self.inner_contours]
        
        # Проверка корректности геометрии
        if not outer_polygon.is_valid:
            raise ValueError("Внешний контур невалиден")
            
        for inner in inner_polygons:
            if not inner.is_valid:
                raise ValueError("Внутренний контур невалиден")
            if not outer_polygon.contains(inner):
                raise ValueError("Внутренний контур выходит за границы внешнего")
        
        # Формирование итоговой геометрии
        self.polygon = outer_polygon
        for inner in inner_polygons:
            self.polygon = self.polygon.difference(inner)
            
        # Расчет геометрических характеристик
        self.area = self.polygon.area
        self.centroid = np.array(self.polygon.centroid.coords[0])
        self.moment_of_inertia = self._calculate_moment_of_inertia()
        
    def _calculate_moment_of_inertia(self):
        """Расчет момента инерции для фигуры"""
        # Упрощенный расчет для плоской фигуры
        return self.area * 100  # Пропорционально площади
        
    def apply_transformation(self, position, angle):
        """
        Применение трансформации к фигуре
        :param position: (x, y) - позиция центра масс
        :param angle: угол поворота в градусах
        """
        # Перемещение центра масс в начало координат
        transformed_outer = self.outer_contour - self.centroid
        # Поворот
        theta = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        transformed_outer = np.dot(transformed_outer, rotation_matrix.T)
        # Перемещение в заданную позицию
        transformed_outer = transformed_outer + np.array(position)
        
        # Аналогичные преобразования для внутренних контуров
        transformed_inners = []
        for contour in self.inner_contours:
            transformed_inner = contour - self.centroid
            transformed_inner = np.dot(transformed_inner, rotation_matrix.T)
            transformed_inner = transformed_inner + np.array(position)
            transformed_inners.append(transformed_inner)
            
        # Создание новой фигуры с преобразованной геометрией
        return PolygonShape(transformed_outer, transformed_inners, self.name)
    
    def get_bounding_box(self):
        """Получение ограничивающего прямоугольника"""
        min_x, min_y, max_x, max_y = self.polygon.bounds
        return (min_x, min_y, max_x, max_y)