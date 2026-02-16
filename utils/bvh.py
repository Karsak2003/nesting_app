import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from collections import deque
import math
import time
import logging

logger = logging.getLogger(__name__)

class BVHNode:
    """
    Узел иерархии ограничивающих объемов (Bounding Volume Hierarchy)
    
    Согласно разделу 2.2.5, BVH дерево является критически важным компонентом
    для ускорения поиска соседних объектов и проверки коллизий в системе ИАГИ.
    """
    
    def __init__(self, bbox: Tuple[float, float, float, float]):
        """
        Инициализация узла BVH
        
        :param bbox: Ограничивающий прямоугольник в формате (min_x, min_y, max_x, max_y)
        """
        self.bbox = bbox  # Ограничивающий прямоугольник узла
        self.left = None  # Левый дочерний узел
        self.right = None  # Правый дочерний узел
        self.objects = []  # Список объектов (для листовых узлов)
        self.is_leaf = True  # Флаг листового узла
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Проверка, содержится ли точка внутри ограничивающего прямоугольника узла
        
        :param point: Точка (x, y)
        :return: True если точка внутри bbox
        """
        x, y = point
        min_x, min_y, max_x, max_y = self.bbox
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def intersects_bbox(self, other_bbox: Tuple[float, float, float, float]) -> bool:
        """
        Проверка пересечения двух ограничивающих прямоугольников
        
        :param other_bbox: Другой прямоугольник (min_x, min_y, max_x, max_y)
        :return: True если прямоугольники пересекаются
        """
        min_x1, min_y1, max_x1, max_y1 = self.bbox
        min_x2, min_y2, max_x2, max_y2 = other_bbox
        
        return not (max_x1 < min_x2 or max_x2 < min_x1 or 
                   max_y1 < min_y2 or max_y2 < min_y1)
    
    def get_center(self) -> Tuple[float, float]:
        """
        Получение центра ограничивающего прямоугольника
        
        :return: Центр (x_center, y_center)
        """
        min_x, min_y, max_x, max_y = self.bbox
        return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    
    def get_surface_area(self) -> float:
        """
        Расчет площади поверхности ограничивающего прямоугольника
        (в 2D - просто площадь)
        
        :return: Площадь bbox
        """
        min_x, min_y, max_x, max_y = self.bbox
        width = max_x - min_x
        height = max_y - min_y
        return width * height if width > 0 and height > 0 else 0.0
    
    def __str__(self):
        return f"BVHNode(bbox={self.bbox}, objects={len(self.objects)}, leaf={self.is_leaf})"


class BVHTree:
    """
    Иерархия ограничивающих объемов (Bounding Volume Hierarchy) для 2D объектов
    
    Реализует функционал, описанный в разделах 2.2.5 и 3.1.2 диссертации:
    - Ускорение поиска соседей с O(n²) до O(log n)
    - Эффективная проверка коллизий в динамической системе ИАГИ
    - Поддержка динамических обновлений для движущихся агентов
    
    Раздел 2.2.5 подчеркивает, что "BVH-деревья позволяют сократить число проверок до O(log n)",
    что критично для масштабируемости системы при n > 100 фигур.
    """
    
    def __init__(self, max_objects_per_leaf: int = 4, 
                 max_tree_depth: int = 20, 
                 rebuild_threshold: float = 0.4):
        """
        Инициализация BVH дерева
        
        :param max_objects_per_leaf: Максимальное число объектов в листовом узле
        :param max_tree_depth: Максимальная глубина дерева
        :param rebuild_threshold: Порог для перестроения дерева (доля переместившихся объектов)
        """
        self.root = None
        self.max_objects_per_leaf = max_objects_per_leaf
        self.max_tree_depth = max_tree_depth
        self.rebuild_threshold = rebuild_threshold
        self.objects = {}  # Хранилище объектов: id -> (bbox, data)
        self.object_count = 0
        self.last_rebuild_time = 0.0
        self.total_rebuilds = 0
    
    def insert(self, bbox: Tuple[float, float, float, float], data: Any, object_id: Optional[int] = None) -> int:
        """
        Вставка объекта в BVH дерево
        
        :param bbox: Ограничивающий прямоугольник объекта (min_x, min_y, max_x, max_y)
        :param data: Произвольные данные объекта
        :param object_id: Идентификатор объекта (если None, генерируется автоматически)
        :return: Идентификатор объекта в дереве
        """
        if object_id is None:
            object_id = self.object_count
            self.object_count += 1
        
        self.objects[object_id] = (bbox, data)
        
        if self.root is None:
            # Создание корневого узла
            self.root = BVHNode(bbox)
            self.root.objects = [object_id]
        else:
            # Обновление bbox корневого узла
            self._expand_root_bbox(bbox)
            # Вставка в существующее дерево
            self._insert_recursive(self.root, bbox, object_id, depth=0)
        
        return object_id
    
    def _expand_root_bbox(self, new_bbox: Tuple[float, float, float, float]):
        """
        Расширение ограничивающего прямоугольника корневого узла
        
        :param new_bbox: Новый прямоугольник для включения
        """
        if self.root is None:
            self.root = BVHNode(new_bbox)
            return
        
        min_x1, min_y1, max_x1, max_y1 = self.root.bbox
        min_x2, min_y2, max_x2, max_y2 = new_bbox
        
        new_min_x = min(min_x1, min_x2)
        new_min_y = min(min_y1, min_y2)
        new_max_x = max(max_x1, max_x2)
        new_max_y = max(max_y1, max_y2)
        
        self.root.bbox = (new_min_x, new_min_y, new_max_x, new_max_y)
    
    def _insert_recursive(self, node: BVHNode, bbox: Tuple[float, float, float, float], 
                         object_id: int, depth: int):
        """
        Рекурсивная вставка объекта в BVH дерево
        
        :param node: Текущий узел для вставки
        :param bbox: Ограничивающий прямоугольник объекта
        :param object_id: Идентификатор объекта
        :param depth: Текущая глубина рекурсии
        """
        if node.is_leaf:
            if len(node.objects) < self.max_objects_per_leaf or depth >= self.max_tree_depth:
                # Добавление в существующий лист
                node.objects.append(object_id)
            else:
                # Разделение листа
                self._split_node(node, depth)
                # Повторная вставка в подходящий дочерний узел
                self._insert_recursive(node, bbox, object_id, depth)
        else:
            # Выбор дочернего узла для вставки по наименьшему расширению bbox
            left_expansion = self._calculate_bbox_expansion(node.left.bbox, bbox)
            right_expansion = self._calculate_bbox_expansion(node.right.bbox, bbox)
            
            if left_expansion <= right_expansion:
                self._expand_bbox(node.left.bbox, bbox)
                self._insert_recursive(node.left, bbox, object_id, depth + 1)
            else:
                self._expand_bbox(node.right.bbox, bbox)
                self._insert_recursive(node.right, bbox, object_id, depth + 1)
    
    def _calculate_bbox_expansion(self, current_bbox: Tuple[float, float, float, float], 
                                 new_bbox: Tuple[float, float, float, float]) -> float:
        """
        Расчет расширения площади bbox при добавлении нового объекта
        
        :param current_bbox: Текущий ограничивающий прямоугольник
        :param new_bbox: Новый прямоугольник для включения
        :return: Дополнительная площадь после расширения
        """
        min_x1, min_y1, max_x1, max_y1 = current_bbox
        min_x2, min_y2, max_x2, max_y2 = new_bbox
        
        current_area = (max_x1 - min_x1) * (max_y1 - min_y1)
        
        new_min_x = min(min_x1, min_x2)
        new_min_y = min(min_y1, min_y2)
        new_max_x = max(max_x1, max_x2)
        new_max_y = max(max_y1, max_y2)
        
        new_area = (new_max_x - new_min_x) * (new_max_y - new_min_y)
        
        return new_area - current_area
    
    def _expand_bbox(self, bbox: Tuple[float, float, float, float], 
                    new_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Расширение bbox для включения нового прямоугольника
        
        :param bbox: Исходный ограничивающий прямоугольник
        :param new_bbox: Новый прямоугольник для включения
        :return: Расширенный прямоугольник
        """
        min_x1, min_y1, max_x1, max_y1 = bbox
        min_x2, min_y2, max_x2, max_y2 = new_bbox
        
        new_min_x = min(min_x1, min_x2)
        new_min_y = min(min_y1, min_y2)
        new_max_x = max(max_x1, max_x2)
        new_max_y = max(max_y1, max_y2)
        
        return (new_min_x, new_min_y, new_max_x, new_max_y)  
    
    def _split_node(self, node: BVHNode, depth: int):
        """
        Разделение узла на два дочерних
        
        :param node: Узел для разделения
        :param depth: Текущая глубина рекурсии
        """
        if len(node.objects) <= 1:
            return
        
        # Выбор оси разделения (по наибольшей длине bbox)
        min_x, min_y, max_x, max_y = node.bbox
        width = max_x - min_x
        height = max_y - min_y
        
        axis = 0 if width >= height else 1  # 0 для x, 1 для y
        
        # Сортировка объектов по центру bbox вдоль выбранной оси
        object_centers = []
        for obj_id in node.objects:
            obj_bbox, _ = self.objects[obj_id]
            center = ((obj_bbox[0] + obj_bbox[2]) / 2.0, (obj_bbox[1] + obj_bbox[3]) / 2.0)
            object_centers.append((obj_id, center[axis]))
        
        # Сортировка по координате центра
        object_centers.sort(key=lambda x: x[1])
        
        # Разделение на две части
        mid = len(object_centers) // 2
        left_objects = object_centers[:mid]
        right_objects = object_centers[mid:]
        
        # Создание дочерних узлов
        left_bbox = self._calculate_bbox_for_objects([obj_id for obj_id, _ in left_objects])
        right_bbox = self._calculate_bbox_for_objects([obj_id for obj_id, _ in right_objects])
        
        node.left = BVHNode(left_bbox)
        node.right = BVHNode(right_bbox)
        
        node.left.objects = [obj_id for obj_id, _ in left_objects]
        node.right.objects = [obj_id for obj_id, _ in right_objects]
        
        node.objects = []
        node.is_leaf = False
    
    def _calculate_bbox_for_objects(self, object_ids: List[int]) -> Tuple[float, float, float, float]:
        """
        Расчет ограничивающего прямоугольника для набора объектов
        
        :param object_ids: Список идентификаторов объектов
        :return: Ограничивающий прямоугольник (min_x, min_y, max_x, max_y)
        """
        if not object_ids:
            return (0, 0, 0, 0)
        
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for obj_id in object_ids:
            bbox, _ = self.objects[obj_id]
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
        
        return (min_x, min_y, max_x, max_y)
    
    def query(self, query_bbox: Tuple[float, float, float, float]) -> List[int]:
        """
        Поиск всех объектов, пересекающихся с заданным прямоугольником
        
        :param query_bbox: Прямоугольник запроса (min_x, min_y, max_x, max_y)
        :return: Список идентификаторов объектов, пересекающихся с запросом
        """
        if self.root is None:
            return []
        
        results = []
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            
            if node.intersects_bbox(query_bbox):
                if node.is_leaf:
                    results.extend(node.objects)
                else:
                    # Добавление дочерних узлов в стек для обработки
                    stack.append(node.left)
                    stack.append(node.right)
        
        return results
    
    def nearest(self, point: Tuple[float, float], max_distance: float = float('inf')) -> Optional[Tuple[int, float]]:
        """
        Поиск ближайшего объекта к заданной точке
        
        :param point: Точка (x, y) для поиска ближайшего объекта
        :param max_distance: Максимальное расстояние для поиска
        :return: Кортеж (object_id, distance) или None если объектов нет в пределах max_distance
        """
        if self.root is None:
            return None
        
        best_id = None
        best_distance = float('inf')
        
        # Использование приоритетной очереди для эффективного поиска
        queue = [(0.0, self.root)]  # (приоритет, узел), приоритет = расстояние до bbox
        
        while queue:
            dist, node = queue.pop(0)
            
            if dist > best_distance or dist > max_distance:
                continue
            
            if node.is_leaf:
                # Проверка расстояния до каждого объекта в листе
                for obj_id in node.objects:
                    obj_bbox, _ = self.objects[obj_id]
                    obj_center = ((obj_bbox[0] + obj_bbox[2]) / 2.0, (obj_bbox[1] + obj_bbox[3]) / 2.0)
                    
                    # Расчет евклидова расстояния между точками
                    distance = math.hypot(point[0] - obj_center[0], point[1] - obj_center[1])
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_id = obj_id
            else:
                # Расчет расстояния до дочерних узлов
                left_dist = self._distance_point_to_bbox(point, node.left.bbox)
                right_dist = self._distance_point_to_bbox(point, node.right.bbox)
                
                # Добавление в очередь с сортировкой по расстоянию
                if left_dist < best_distance and left_dist < max_distance:
                    queue.append((left_dist, node.left))
                if right_dist < best_distance and right_dist < max_distance:
                    queue.append((right_dist, node.right))
                
                # Сортировка очереди по расстоянию
                queue.sort(key=lambda x: x[0])
        
        if best_id is not None and best_distance <= max_distance:
            return (best_id, best_distance)
        return None
    
    def _distance_point_to_bbox(self, point: Tuple[float, float], 
                               bbox: Tuple[float, float, float, float]) -> float:
        """
        Расчет расстояния от точки до ограничивающего прямоугольника
        
        :param point: Точка (x, y)
        :param bbox: Ограничивающий прямоугольник (min_x, min_y, max_x, max_y)
        :return: Минимальное расстояние от точки до прямоугольника
        """
        x, y = point
        min_x, min_y, max_x, max_y = bbox
        
        # Проверка, находится ли точка внутри прямоугольника
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return 0.0
        
        # Расчет расстояния до ближайшей границы
        dx = max(min_x - x, 0, x - max_x)
        dy = max(min_y - y, 0, y - max_y)
        
        return math.hypot(dx, dy)
    
    def update_object(self, object_id: int, new_bbox: Tuple[float, float, float, float]) -> bool:
        """
        Обновление положения объекта в BVH дереве
        
        :param object_id: Идентификатор объекта
        :param new_bbox: Новый ограничивающий прямоугольник
        :return: True если обновление успешно
        """
        if object_id not in self.objects:
            return False
        
        old_bbox, data = self.objects[object_id]
        self.objects[object_id] = (new_bbox, data)
        
        # Простая стратегия: если объект сильно переместился, перестроить дерево
        if self._bbox_movement_significant(old_bbox, new_bbox):
            self.rebuild()
            return True
        
        # Иначе попытаться обновить локально
        return self._update_object_recursive(self.root, object_id, new_bbox)
    
    def _bbox_movement_significant(self, old_bbox: Tuple[float, float, float, float], 
                                   new_bbox: Tuple[float, float, float, float]) -> bool:
        """
        Проверка, является ли перемещение bbox значительным
        
        :param old_bbox: Старый ограничивающий прямоугольник
        :param new_bbox: Новый ограничивающий прямоугольник
        :return: True если перемещение значительное
        """
        # Расчет центров
        old_center = ((old_bbox[0] + old_bbox[2]) / 2.0, (old_bbox[1] + old_bbox[3]) / 2.0)
        new_center = ((new_bbox[0] + new_bbox[2]) / 2.0, (new_bbox[1] + new_bbox[3]) / 2.0)
        
        # Расчет расстояния между центрами
        movement = math.hypot(old_center[0] - new_center[0], old_center[1] - new_center[1])
        
        # Сравнение с размером bbox
        old_size = math.hypot(old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1])
        
        return movement > 0.5 * old_size  # Если переместился больше чем на половину своего размера
    
    def _update_object_recursive(self, node: BVHNode, object_id: int, 
                                new_bbox: Tuple[float, float, float, float]) -> bool:
        """
        Рекурсивное обновление положения объекта в BVH дереве
        
        :param node: Текущий узел
        :param object_id: Идентификатор объекта
        :param new_bbox: Новый ограничивающий прямоугольник
        :return: True если обновление успешно
        """
        if object_id not in node.objects:
            return False
        
        # Обновление bbox узла
        node.bbox = self._expand_bbox(node.bbox, new_bbox)
        
        if node.is_leaf:
            return True
        
        # Попытка обновить в дочерних узлах
        left_updated = self._update_object_recursive(node.left, object_id, new_bbox)
        right_updated = self._update_object_recursive(node.right, object_id, new_bbox)
        
        if left_updated or right_updated:
            return True
        
        # Если объект не найден в дочерних узлах, возможно, его нужно переместить
        # В этом случае лучше перестроить дерево
        return False
    
    def rebuild(self):
        """
        Полное перестроение BVH дерева
        
        Согласно разделу 2.4.8, перестроение дерева критично для масштабируемости
        при работе с динамически изменяющимися объектами в системе ИАГИ.
        """
        start_time = time.time()
        
        if not self.objects:
            self.root = None
            return
        
        # Сбор всех объектов для перестроения
        all_objects = [(obj_id, bbox) for obj_id, (bbox, _) in self.objects.items()]
        
        # Построение нового дерева
        self.root = self._build_tree_recursive(all_objects, depth=0)
        
        self.last_rebuild_time = time.time() - start_time
        self.total_rebuilds += 1
        
        logger.debug(f"BVH дерево перестроено за {self.last_rebuild_time:.4f} секунд. "
                    f"Объектов: {len(self.objects)}, Перестроений: {self.total_rebuilds}")
    
    def _build_tree_recursive(self, objects: List[Tuple[int, Tuple[float, float, float, float]]], 
                             depth: int) -> BVHNode:
        """
        Рекурсивное построение BVH дерева
        
        :param objects: Список объектов в формате (object_id, bbox)
        :param depth: Текущая глубина рекурсии
        :return: Корневой узел построенного поддерева
        """
        if not objects:
            return BVHNode((0, 0, 0, 0))
        
        # Расчет bbox для текущего узла
        bbox = self._calculate_bbox_for_objects([obj_id for obj_id, _ in objects])
        node = BVHNode(bbox)
        
        # Проверка условий для листового узла
        if len(objects) <= self.max_objects_per_leaf or depth >= self.max_tree_depth:
            node.objects = [obj_id for obj_id, _ in objects]
            node.is_leaf = True
            return node
        
        # Выбор оси разделения
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        
        axis = 0 if width >= height else 1  # 0 для x, 1 для y
        
        # Сортировка объектов по центру bbox вдоль выбранной оси
        object_centers = []
        for obj_id, bbox in objects:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            object_centers.append((obj_id, bbox, center[axis]))
        
        # Сортировка по координате центра
        object_centers.sort(key=lambda x: x[2])
        
        # Разделение на две части
        mid = len(object_centers) // 2
        left_objects = object_centers[:mid]
        right_objects = object_centers[mid:]
        
        # Рекурсивное построение дочерних узлов
        left_node = self._build_tree_recursive([(obj_id, bbox) for obj_id, bbox, _ in left_objects], depth + 1)
        right_node = self._build_tree_recursive([(obj_id, bbox) for obj_id, bbox, _ in right_objects], depth + 1)
        
        node.left = left_node
        node.right = right_node
        node.is_leaf = False
        
        return node
    
    def remove(self, object_id: int) -> bool:
        """
        Удаление объекта из BVH дерева
        
        :param object_id: Идентификатор объекта
        :return: True если удаление успешно
        """
        if object_id not in self.objects:
            return False
        
        del self.objects[object_id]
        
        # После удаления объекта лучше перестроить дерево для сохранения баланса
        if len(self.objects) > 0:
            self.rebuild()
        else:
            self.root = None
        
        return True
    
    def get_object_count(self) -> int:
        """
        Получение количества объектов в дереве
        
        :return: Число объектов
        """
        return len(self.objects)
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """
        Получение статистики о BVH дереве
        
        :return: Словарь со статистикой
        """
        if self.root is None:
            return {
                'total_objects': 0,
                'tree_depth': 0,
                'total_nodes': 0,
                'leaf_nodes': 0,
                'avg_objects_per_leaf': 0.0,
                'last_rebuild_time': self.last_rebuild_time,
                'total_rebuilds': self.total_rebuilds
            }
        
        # Обход дерева для сбора статистики
        queue = deque([(self.root, 0)])  # (node, depth)
        max_depth = 0
        total_nodes = 0
        leaf_nodes = 0
        total_objects_in_leaves = 0
        
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            total_nodes += 1
            
            if node.is_leaf:
                leaf_nodes += 1
                total_objects_in_leaves += len(node.objects)
            else:
                if node.left:
                    queue.append((node.left, depth + 1))
                if node.right:
                    queue.append((node.right, depth + 1))
        
        return {
            'total_objects': len(self.objects),
            'tree_depth': max_depth,
            'total_nodes': total_nodes,
            'leaf_nodes': leaf_nodes,
            'avg_objects_per_leaf': total_objects_in_leaves / leaf_nodes if leaf_nodes > 0 else 0.0,
            'last_rebuild_time': self.last_rebuild_time,
            'total_rebuilds': self.total_rebuilds
        }
    
    def visualize(self, filename: str):
        """
        Визуализация BVH дерева (только для 2D)
        
        :param filename: Имя файла для сохранения изображения
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Получение границ всего дерева
            min_x, min_y, max_x, max_y = self.root.bbox
            
            # Отступы для визуализации
            padding = 0.1
            width = max_x - min_x
            height = max_y - min_y
            min_x -= width * padding
            max_x += width * padding
            min_y -= height * padding
            max_y += height * padding
            
            # Визуализация узлов дерева
            queue = [self.root]
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
            
            while queue:
                node = queue.pop(0)
                min_x_n, min_y_n, max_x_n, max_y_n = node.bbox
                
                # Выбор цвета в зависимости от глубины (грубая оценка)
                depth = self._get_node_depth(self.root, node)
                color = colors[depth % len(colors)]
                
                # Рисование прямоугольника
                rect = Rectangle((min_x_n, min_y_n), max_x_n - min_x_n, max_y_n - min_y_n,
                               fill=False, color=color, linewidth=1.0 - depth * 0.1)
                ax.add_patch(rect)
                
                # Добавление дочерних узлов в очередь
                if not node.is_leaf:
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
            
            # Визуализация объектов
            for obj_id, (bbox, _) in self.objects.items():
                min_x_o, min_y_o, max_x_o, max_y_o = bbox
                rect = Rectangle((min_x_o, min_y_o), max_x_o - min_x_o, max_y_o - min_y_o,
                               fill=True, color='orange', alpha=0.3)
                ax.add_patch(rect)
                # Подпись с ID объекта
                ax.text((min_x_o + max_x_o) / 2, (min_y_o + max_y_o) / 2, str(obj_id),
                       ha='center', va='center', fontsize=8, color='black')
            
            # Настройка осей
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            ax.set_title('BVH Tree Visualization')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"BVH дерево визуализировано и сохранено в {filename}")
            return True
        
        except ImportError:
            logger.error("Для визуализации BVH дерева требуется matplotlib. Установите: pip install matplotlib")
            return False
    
    def _get_node_depth(self, root: BVHNode, target: BVHNode) -> int:
        """
        Получение глубины узла в дереве (для визуализации)
        
        :param root: Корневой узел
        :param target: Целевой узел
        :return: Глубина узла
        """
        if root is None or root == target:
            return 0
        
        if not root.is_leaf:
            left_depth = self._get_node_depth(root.left, target)
            if left_depth >= 0:
                return left_depth + 1
            
            right_depth = self._get_node_depth(root.right, target)
            if right_depth >= 0:
                return right_depth + 1
        
        return -1
    
    def __str__(self):
        """Строковое представление для отладки"""
        stats = self.get_tree_stats()
        return (f"BVHTree(objects={stats['total_objects']}, depth={stats['tree_depth']}, "
                f"nodes={stats['total_nodes']}, rebuilds={stats['total_rebuilds']})")


def create_bvh_from_shapes(shapes: List[Any], max_objects_per_leaf: int = 4) -> BVHTree:
    """
    Создание BVH дерева из списка фигур
    
    :param shapes: Список фигур, поддерживающих метод get_bounding_box()
    :param max_objects_per_leaf: Максимальное число объектов в листе
    :return: Построенное BVH дерево
    """
    bvh = BVHTree(max_objects_per_leaf=max_objects_per_leaf)
    
    for i, shape in enumerate(shapes):
        if hasattr(shape, 'get_bounding_box'):
            bbox = shape.get_bounding_box()
        elif hasattr(shape, 'polygon') and hasattr(shape.polygon, 'bounds'):
            # Для Shapely геометрии
            min_x, min_y, max_x, max_y = shape.polygon.bounds
            bbox = (min_x, min_y, max_x, max_y)
        else:
            raise ValueError(f"Фигура {i} не поддерживает получение ограничивающего прямоугольника")
        
        bvh.insert(bbox, shape, object_id=i)
    
    return bvh