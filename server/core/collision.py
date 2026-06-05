import numpy as np
from typing import Tuple, Any, Optional

from utils.bvh import BVHTree
from utils.sdf import compute_sdf_field

class CollisionDetector:
    """
    Система обнаружения столкновений на основе SDF и BVH
    """
    def __init__(self, resolution: float = 0.5):
        """
        Инициализация детектора столкновений
        :param resolution: разрешение для SDF поля в мм
        """
        self.resolution = resolution
        self.bvh_tree: Optional[BVHTree] = None
        self.sdf_fields: dict = {}
        
    def build_spatial_index(self, shapes: list, sheet_size: Tuple[float, float]) -> None:
        """
        Построение пространственного индекса для ускорения поиска
        :param shapes: список фигур на листе
        :param sheet_size: размеры листа (width, height)
        """
        self.bvh_tree = BVHTree()
        for i, shape in enumerate(shapes):
            bbox = shape.get_bounding_box()
            self.bvh_tree.insert(bbox, i)
        
        # Предварительное вычисление SDF полей для каждой уникальной геометрии
        for shape in shapes:
            shape_id = id(shape)
            if shape_id not in self.sdf_fields:
                self.sdf_fields[shape_id] = compute_sdf_field(
                    shape.polygon, 
                    resolution=self.resolution,
                    bounds=(0, 0, sheet_size[0], sheet_size[1])
                )
    
    def find_nearby_shapes(
        self, 
        shape: Any, 
        shapes: list, 
        radius: float
    ) -> list:
        """
        Поиск фигур в заданном радиусе
        :param shape: целевая фигура
        :param shapes: все фигуры на листе
        :param radius: радиус поиска
        :return: список индексов соседних фигур
        """
        bbox = shape.get_bounding_box()
        search_box = (
            bbox[0] - radius, bbox[1] - radius,
            bbox[2] + radius, bbox[3] + radius
        )
        return self.bvh_tree.query(search_box)
    
    def calculate_min_distance(
        self, 
        shape1: Any, 
        shape2: Any, 
        min_gap: float = 0.0
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Расчет минимального расстояния между двумя фигурами с учетом минимального зазора
        :param shape1, shape2: фигуры
        :param min_gap: минимальный технологический зазор
        :return: (distance, closest_point1, closest_point2, normal_vector)
        """
        distance = shape1.polygon.distance(shape2.polygon)

        if shape1.polygon.intersects(shape2.polygon):
            intersection = shape1.polygon.intersection(shape2.polygon)
            if intersection.area > 0:
                try:
                    clearance1 = shape1.polygon.minimum_clearance
                    clearance2 = shape2.polygon.minimum_clearance
                    penetration = min(clearance1, clearance2, intersection.area ** 0.5)
                    distance = -penetration
                except:
                    distance = -(intersection.area ** 0.5)

        effective_distance = distance - min_gap

        if distance > 0:
            from shapely.ops import nearest_points
            try:
                pt1, pt2 = nearest_points(shape1.polygon, shape2.polygon)
                closest_point1 = np.array([pt1.x, pt1.y])
                closest_point2 = np.array([pt2.x, pt2.y])
            except:
                closest_point1 = shape1.centroid
                closest_point2 = shape2.centroid
        else:
            closest_point1 = shape1.centroid
            closest_point2 = shape2.centroid
        
        direction = closest_point2 - closest_point1
        norm = np.linalg.norm(direction)
        
        if norm > 1e-6:
            normal_vector = direction / norm
        else:
            centroid_dir = shape2.centroid - shape1.centroid
            centroid_norm = np.linalg.norm(centroid_dir)
            if centroid_norm > 1e-6:
                normal_vector = centroid_dir / centroid_norm
            else:
                normal_vector = np.array([1.0, 0.0])
        
        return effective_distance, closest_point1, closest_point2, normal_vector
    
    def check_collision(self, shape1: Any, shape2: Any, min_gap: float = 0.0) -> bool:
        """
        Проверка столкновения с учетом минимального зазора
        :return: True если есть столкновение (расстояние < 0 с учетом зазора)
        """
        distance, _, _, _ = self.calculate_min_distance(shape1, shape2, min_gap)
        return distance < 0