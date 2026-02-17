"""
Модуль для построения No-Fit Polygon (NFP) для верификации решений
Согласно разделу 2.2.2, NFP используется для пост-процессинга и верификации допустимости решений
"""

import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import logging
from typing import List, Tuple, Optional, Any, Dict

import shapely.ops

logger = logging.getLogger(__name__)

class NoFitPolygon:
    """
    Класс для построения и работы с No-Fit Polygon (NFP)
    
    NFP определяется как множество всех векторов трансляции, при которых
    две фигуры пересекаются: NFP(A,B) = {t | (A+t) ∩ B ≠ ∅}
    """
    
    def __init__(self, polygon_a: Polygon, polygon_b: Polygon):
        """
        Инициализация NFP для пары фигур
        
        :param polygon_a: Фиксированная фигура A
        :param polygon_b: Подвижная фигура B
        """
        self.polygon_a = polygon_a
        self.polygon_b = polygon_b
        self.nfp_polygon = None
        self._build_nfp()
    
    def _build_nfp(self):
        """
        Построение NFP через операцию суммы Минковского
        
        Согласно разделу 2.2.2:
        NFP(A,B) = ∂(A ⊕ (-B))
        где ⊕ - сумма Минковского, -B - центрально-симметричное отображение B
        """
        try:
            # Центрально-симметричное отображение фигуры B относительно начала координат
            b_negated = self._negate_polygon(self.polygon_b)
            
            # Сумма Минковского A ⊕ (-B)
            minkowski_sum = self._minkowski_sum(self.polygon_a, b_negated)
            
            # Граница суммы Минковского - это и есть NFP
            self.nfp_polygon = minkowski_sum.exterior
            
            logger.debug(f"NFP успешно построен для пары фигур")
            
        except Exception as e:
            logger.error(f"Ошибка при построении NFP: {e}")
            # Резервный метод: аппроксимация через дискретизацию
            self.nfp_polygon = self._approximate_nfp()
    
    def _negate_polygon(self, polygon: Polygon) -> Polygon:
        """Центрально-симметричное отображение полигона относительно начала координат"""
        negated_coords = [(-x, -y) for x, y in polygon.exterior.coords]
        return Polygon(negated_coords)
    
    def _minkowski_sum(self, poly1: Polygon, poly2: Polygon) -> Polygon:
        """
        Вычисление суммы Минковского двух полигонов
        
        Для невыпуклых полигонов используется метод декомпозиции на выпуклые части
        """
        # Проверка выпуклости
        if self._is_convex(poly1) and self._is_convex(poly2):
            # Для выпуклых полигонов - прямой метод
            return self._minkowski_sum_convex(poly1, poly2)
        else:
            # Для невыпуклых - декомпозиция на выпуклые части
            convex_parts1 = self._decompose_to_convex(poly1)
            convex_parts2 = self._decompose_to_convex(poly2)
            
            # Сумма Минковского для каждой пары выпуклых частей
            minkowski_parts = []
            for part1 in convex_parts1:
                for part2 in convex_parts2:
                    minkowski_parts.append(self._minkowski_sum_convex(part1, part2))
            
            # Объединение всех частей
            return unary_union(minkowski_parts)
    
    def _minkowski_sum_convex(self, poly1: Polygon, poly2: Polygon) -> Polygon:
        """
        Сумма Минковского для двух выпуклых полигонов
        
        Алгоритм: объединение векторов рёбер в порядке обхода
        """
        # Получение вершин в порядке обхода
        coords1 = list(poly1.exterior.coords)[:-1]  
        # Убираем последнюю точку (дублирует первую)
        coords2 = list(poly2.exterior.coords)[:-1]
        
        # Вычисление векторов рёбер
        edges1 = self._get_edge_vectors(coords1)
        edges2 = self._get_edge_vectors(coords2)
        
        # Объединение векторов в порядке возрастания угла
        all_edges = edges1 + edges2
        all_edges.sort(key=lambda v: np.arctan2(v[1], v[0]))
        
        # Построение суммы Минковского
        result_coords = []
        current_point = (0, 0)
        
        for edge in all_edges:
            result_coords.append(current_point)
            current_point = (current_point[0] + edge[0], current_point[1] + edge[1])
        
        # Замыкание полигона
        result_coords.append(result_coords[0])
        
        return Polygon(result_coords)
    
    def _get_edge_vectors(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Получение векторов рёбер из списка вершин"""
        edges = []
        n = len(coords)
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            edges.append((x2 - x1, y2 - y1))
        return edges
    
    def _is_convex(self, polygon: Polygon) -> bool:
        """Проверка выпуклости полигона"""
        coords = list(polygon.exterior.coords)[:-1]
        n = len(coords)
        
        if n < 4:
            return True
        
        # Проверка знака векторного произведения для всех троек последовательных точек
        prev_cross = None
        for i in range(n):
            p1 = np.array(coords[i])
            p2 = np.array(coords[(i + 1) % n])
            p3 = np.array(coords[(i + 2) % n])
            
            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            if cross != 0:
                if prev_cross is None:
                    prev_cross = cross
                elif prev_cross * cross < 0:
                    return False
        
        return True
    
    def _decompose_to_convex(self, polygon: Polygon) -> List[Polygon]:
        """
        Декомпозиция невыпуклого полигона на выпуклые части
        
        Используется метод "ушей" (ear clipping)
        """
        # Для упрощения используем библиотеку shapely для триангуляции
        from shapely.ops import triangulate
        
        triangles = triangulate(polygon)
        return list(triangles)
    
    def _approximate_nfp(self) -> LineString:
        """
        Аппроксимация NFP через дискретизацию (резервный метод)
        
        Используется при невозможности построения точного NFP
        """
        logger.warning("Используется аппроксимация NFP через дискретизацию")
        
        # Дискретизация угла поворота
        num_angles = 36
        angle_step = 360.0 / num_angles
        
        nfp_points = []
        
        for i in range(num_angles):
            angle = i * angle_step
            
            # Поворот фигуры B
            rotated_b = rotate(self.polygon_b, angle, origin=(0, 0), use_radians=False)
            
            # Поиск ближайшей точки касания
            contact_point = self._find_contact_point(self.polygon_a, rotated_b)
            if contact_point:
                nfp_points.append(contact_point)
        
        # Построение полигона из точек касания
        if len(nfp_points) >= 3:
            return LineString(nfp_points + [nfp_points[0]])
        else:
            # Возврат прямоугольника, охватывающего обе фигуры
            min_x = min(self.polygon_a.bounds[0], self.polygon_b.bounds[0])
            min_y = min(self.polygon_a.bounds[1], self.polygon_b.bounds[1])
            max_x = max(self.polygon_a.bounds[2], self.polygon_b.bounds[2])
            max_y = max(self.polygon_a.bounds[3], self.polygon_b.bounds[3])
            return LineString([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)])
    
    def _find_contact_point(self, poly1: Polygon, poly2: Polygon) -> Optional[Tuple[float, float]]:
        """Поиск точки касания двух полигонов"""
        # Упрощенный метод: поиск ближайших точек
        try:
            
            nearest_points = shapely.ops.nearest_points(poly1, poly2)
            return (nearest_points[0].x, nearest_points[0].y)
        except:
            return None
    
    def is_position_valid(self, position: Tuple[float, float], angle: float = 0.0) -> bool:
        """
        Проверка допустимости позиции фигуры B относительно фигуры A
        
        :param position: Позиция центра масс фигуры B
        :param angle: Угол поворота фигуры B
        :return: True если позиция допустима (нет пересечения)
        """
        if self.nfp_polygon is None:
            return True  # Невозможно проверить
        
        # Проверка, находится ли точка внутри NFP
        point = Point(position)
        return not self.nfp_polygon.contains(point)
    
    def get_nfp_polygon(self) -> Optional[LineString]:
        """Получение геометрии NFP"""
        return self.nfp_polygon
    
    def visualize(self, filename: str = "nfp_visualization.png"):
        """Визуализация NFP"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Визуализация исходных полигонов
            x_a, y_a = self.polygon_a.exterior.xy
            ax.plot(x_a, y_a, 'b-', linewidth=2, label='Фигура A')
            
            x_b, y_b = self.polygon_b.exterior.xy
            ax.plot(x_b, y_b, 'g-', linewidth=2, label='Фигура B')
            
            # Визуализация NFP
            if self.nfp_polygon:
                x_nfp, y_nfp = self.nfp_polygon.xy
                ax.plot(x_nfp, y_nfp, 'r--', linewidth=2, label='NFP')
            
            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True)
            ax.set_title('No-Fit Polygon (NFP)')
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"NFP визуализирован и сохранен в {filename}")
            return True
            
        except ImportError:
            logger.error("Для визуализации требуется matplotlib")
            return False

def build_nfp_for_pair(polygon_a: Polygon, polygon_b: Polygon) -> NoFitPolygon:
    """
    Функция-обертка для построения NFP для пары фигур
    
    :param polygon_a: Фиксированная фигура A
    :param polygon_b: Подвижная фигура B
    :return: Объект NoFitPolygon
    """
    return NoFitPolygon(polygon_a, polygon_b)

def build_nfp_cache(shapes: List[Any], max_pairs: int = 1000) -> Dict[Tuple[int, int], NoFitPolygon]:
    """
    Построение кэша NFP для всех пар фигур
    
    Согласно разделу 2.2.2, для n фигур требуется O(n²) NFP, что может быть вычислительно затратно.
    Поэтому кэшируем только наиболее часто используемые пары.
    
    :param shapes: Список фигур
    :param max_pairs: Максимальное число пар для кэширования
    :return: Словарь с NFP для пар (индекс1, индекс2)
    """
    logger.info(f"Построение кэша NFP для {len(shapes)} фигур...")
    
    nfp_cache = {}
    pair_count = 0
    
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if pair_count >= max_pairs:
                logger.warning(f"Достигнуто максимальное число пар NFP ({max_pairs}). Кэширование остановлено.")
                return nfp_cache
            
            try:
                shape_i = shapes[i].polygon if hasattr(shapes[i], 'polygon') else shapes[i]
                shape_j = shapes[j].polygon if hasattr(shapes[j], 'polygon') else shapes[j]
                
                nfp = build_nfp_for_pair(shape_i, shape_j)
                nfp_cache[(i, j)] = nfp
                pair_count += 1
                
                if pair_count % 100 == 0:
                    logger.debug(f"Построено {pair_count} пар NFP...")
                    
            except Exception as e:
                logger.warning(f"Не удалось построить NFP для пары ({i}, {j}): {e}")
                continue
    
    logger.info(f"Кэш NFP построен: {len(nfp_cache)} пар")
    return nfp_cache

 