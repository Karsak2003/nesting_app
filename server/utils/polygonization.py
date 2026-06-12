import numpy as np
import math
from typing import List, Tuple, Optional, Union, Dict, Any
from core.geometry import PolygonShape
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
import time
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

class AdaptivePolygonizer:
    """
    Класс для адаптивной полигонализации криволинейных геометрических объектов
    
    Реализует функционал, описанный в разделе 2.1.7 диссертации:
    - Адаптивная полигонализация NURBS-кривых с точностью до 0.1 мм
    - Уменьшение числа вершин на 40-60% по сравнению с равномерной дискретизацией
    - Сохранение погрешности плотности упаковки менее 0.8%
    - Поддержка сложных геометрий с отверстиями и составными контурами
    """
    
    def __init__(self, tolerance: float = 0.1, max_recursion_depth: int = 10, 
                 max_vertices: int = 200, preserve_topology: bool = True):
        """
        Инициализация адаптивного полигонализатора
        
        :param tolerance: Допустимая погрешность аппроксимации в мм
        :param max_recursion_depth: Максимальная глубина рекурсии при разбиении
        :param max_vertices: Максимальное число вершин в результирующем полигоне
        :param preserve_topology: Сохранять ли топологию (отверстия, связность)
        """
        self.tolerance = tolerance
        self.max_recursion_depth = max_recursion_depth
        self.max_vertices = max_vertices
        self.preserve_topology = preserve_topology
        
        # Параметры для разных типов производства (раздел 2.1.7)
        self.tolerance_profiles = {
            'high_precision': 0.05,    # Авиация, космос
            'medium_precision': 0.1,   # Машиностроение
            'low_precision': 0.5       # Текстиль, картон
        }
        
        # Статистика для анализа производительности
        self.stats = {
            'total_curves_processed': 0,
            'total_vertices_generated': 0,
            'processing_time': 0.0,
            'reduction_ratio': 0.0
        }
    
    def set_tolerance_profile(self, profile_name: str):
        """
        Установка профиля точности в соответствии с классом производства
        
        :param profile_name: 'high_precision', 'medium_precision' или 'low_precision'
        """
        if profile_name in self.tolerance_profiles:
            self.tolerance = self.tolerance_profiles[profile_name]
            logger.info(f"Установлен профиль точности '{profile_name}' с допуском {self.tolerance} мм")
        else:
            raise ValueError(f"Неизвестный профиль точности: {profile_name}. "
                           f"Доступные: {list(self.tolerance_profiles.keys())}")
    
    def polygonize_curve(self, curve_type: str, curve_data: Any, 
                        start_param: float = 0.0, end_param: float = 1.0) -> List[Tuple[float, float]]:
        """
        Полигонализация кривой заданного типа
        
        :param curve_type: Тип кривой ('nurbs', 'circle', 'ellipse', 'bezier', 'spline')
        :param curve_data: Данные кривой (зависит от типа)
        :param start_param: Начальный параметр кривой
        :param end_param: Конечный параметр кривой
        :return: Список вершин полигональной аппроксимации
        """
        start_time = time.time()
        
        try:
            if curve_type.lower() == 'nurbs':
                vertices = self._polygonize_nurbs(curve_data, start_param, end_param)
            elif curve_type.lower() == 'circle':
                vertices = self._polygonize_circle(curve_data, start_param, end_param)
            elif curve_type.lower() == 'ellipse':
                vertices = self._polygonize_ellipse(curve_data, start_param, end_param)
            elif curve_type.lower() == 'bezier':
                vertices = self._polygonize_bezier(curve_data, start_param, end_param)
            elif curve_type.lower() == 'spline':
                vertices = self._polygonize_spline(curve_data, start_param, end_param)
            else:
                raise ValueError(f"Неизвестный тип кривой: {curve_type}")
            
            # Постобработка: удаление дубликатов и проверка числа вершин
            vertices = self._postprocess_vertices(vertices)
            
            # Обновление статистики
            processing_time = time.time() - start_time
            self.stats['total_curves_processed'] += 1
            self.stats['total_vertices_generated'] += len(vertices)
            self.stats['processing_time'] += processing_time
            
            logger.debug(f"Полигонализация {curve_type} завершена. Вершин: {len(vertices)}, время: {processing_time:.4f}с")
            
            return vertices
            
        except Exception as e:
            logger.error(f"Ошибка при полигонализации кривой {curve_type}: {e}")
            raise
    
    def _polygonize_nurbs(self, nurbs_data: Dict[str, Any], 
                          start_param: float = 0.0, end_param: float = 1.0) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация NURBS-кривой
        
        Согласно разделу 2.1.7, NURBS являются стандартом обмена данными ISO 10303-21,
        и их адаптивная аппроксимация критична для промышленного применения.
        
        Алгоритм:
        1. Рекурсивное разбиение кривой на сегменты
        2. Оценка отклонения от хорды для каждого сегмента
        3. Добавление вершин только там, где отклонение превышает допуск
        """
        control_points = np.array(nurbs_data['control_points'])
        weights = nurbs_data.get('weights', np.ones(len(control_points)))
        knots = nurbs_data.get('knots')
        degree = nurbs_data.get('degree', 3)
        
        # Рекурсивная адаптивная полигонализация
        vertices = []
        self._recursive_nurbs_subdivision(
            control_points, weights, knots, degree,
            start_param, end_param, vertices, 0
        )
        
        return vertices
    
    def _recursive_nurbs_subdivision(self, control_points: np.ndarray, weights: np.ndarray,
                                    knots: Optional[np.ndarray], degree: int,
                                    t_start: float, t_end: float, vertices: List[Tuple[float, float]],
                                    depth: int):
        """
        Рекурсивное разбиение NURBS-кривой для адаптивной полигонализации
        
        :param control_points: Контрольные точки NURBS
        :param weights: Веса контрольных точек
        :param knots: Узловой вектор
        :param degree: Степень сплайна
        :param t_start, t_end: Параметрические границы сегмента
        :param vertices: Список вершин для заполнения
        :param depth: Текущая глубина рекурсии
        """
        if depth > self.max_recursion_depth or len(vertices) > self.max_vertices:
            return
        
        # Вычисление точек на кривой
        t_mid = (t_start + t_end) / 2.0
        p_start = self._evaluate_nurbs_point(control_points, weights, knots, degree, t_start)
        p_mid = self._evaluate_nurbs_point(control_points, weights, knots, degree, t_mid)
        p_end = self._evaluate_nurbs_point(control_points, weights, knots, degree, t_end)
        
        # Вычисление расстояния от средней точки до хорды
        chord = LineString([p_start, p_end])
        distance = chord.distance(Point(p_mid))
        
        # Если отклонение меньше допуска или сегмент достаточно мал - добавляем вершины
        if distance < self.tolerance or (t_end - t_start) < 0.01:
            if not vertices or not np.array_equal(vertices[-1], p_start):
                vertices.append(p_start)
            vertices.append(p_end)
        else:
            # Рекурсивное разбиение
            self._recursive_nurbs_subdivision(
                control_points, weights, knots, degree,
                t_start, t_mid, vertices, depth + 1
            )
            self._recursive_nurbs_subdivision(
                control_points, weights, knots, degree,
                t_mid, t_end, vertices, depth + 1
            )
    
    def _evaluate_nurbs_point(self, control_points: np.ndarray, weights: np.ndarray,
                             knots: Optional[np.ndarray], degree: int, t: float) -> Tuple[float, float]:
        """
        Вычисление точки на NURBS-кривой для заданного параметра t
        """
        # Для упрощения используем scipy для интерполяции
        # В реальной промышленной реализации следует использовать специализированные NURBS-библиотеки
        try:
            from scipy.interpolate import BSpline
        except ImportError:
            logger.warning("scipy не установлен. Используется упрощенная аппроксимация NURBS.")
            # Упрощенная линейная интерполяция
            idx = int(t * (len(control_points) - 1))
            return control_points[idx].tolist()
        
        # Создание B-сплайна
        if knots is None:
            knots = np.linspace(0, 1, len(control_points) - degree + 1)
            knots = np.concatenate((
                np.zeros(degree),
                knots,
                np.ones(degree)
            ))
        
        # Вычисление точки
        spline_x = BSpline(knots, control_points[:, 0] * weights, degree)
        spline_y = BSpline(knots, control_points[:, 1] * weights, degree)
        denom = BSpline(knots, weights, degree)(t)
        
        if abs(denom) < 1e-10:
            denom = 1e-10
        
        x = spline_x(t) / denom
        y = spline_y(t) / denom
        
        return (x, y)
    
    def _polygonize_circle(self, circle_data: Dict[str, Any], 
                          start_angle: float = 0.0, end_angle: float = 2*math.pi) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация окружности
        
        Для окружности можно вычислить оптимальное число сегментов аналитически,
        основываясь на допустимом отклонении от дуги.
        """
        center = np.array(circle_data['center'])
        radius = circle_data['radius']
        
        # Расчет минимального числа сегментов для заданной точности
        # Формула: max_error = radius * (1 - cos(theta/2)), где theta = 2*pi/n
        # Решаем относительно n
        if radius < self.tolerance:
            return [tuple(center), tuple(center)]  # Вырожденный случай
        
        theta_max = 2 * math.acos(1 - self.tolerance / radius)
        n_segments = max(8, int(2 * math.pi / theta_max))  # Минимум 8 сегментов
        
        # Корректировка для частичных окружностей
        angle_range = end_angle - start_angle
        n_segments = max(4, int(n_segments * angle_range / (2 * math.pi)))
        
        vertices = []
        for i in range(n_segments + 1):
            angle = start_angle + angle_range * i / n_segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        
        return vertices
    
    def _polygonize_ellipse(self, ellipse_data: Dict[str, Any],
                          start_param: float = 0.0, end_param: float = 2*math.pi) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация эллипса
        
        Эллипс параметрически представляется как:
        x = cx + a*cos(t)*cos(phi) - b*sin(t)*sin(phi)
        y = cy + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)
        где a, b - полуоси, phi - угол поворота
        """
        center = np.array(ellipse_data['center'])
        a = ellipse_data['major_radius']  # Большая полуось
        b = ellipse_data['minor_radius']  # Малая полуось
        phi = ellipse_data.get('rotation', 0.0)  # Угол поворота
        
        # Расчет максимальной кривизны для определения числа сегментов
        max_curvature = max(1/a, 1/b)
        theta_max = 2 * math.acos(1 - self.tolerance * max_curvature)
        n_segments = max(8, int(2 * math.pi / theta_max))
        
        # Корректировка для частичных эллипсов
        param_range = end_param - start_param
        n_segments = max(4, int(n_segments * param_range / (2 * math.pi)))
        
        vertices = []
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        for i in range(n_segments + 1):
            t = start_param + param_range * i / n_segments
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            
            x = (center[0] + a * cos_t * cos_phi - b * sin_t * sin_phi)
            y = (center[1] + a * cos_t * sin_phi + b * sin_t * cos_phi)
            vertices.append((x, y))
        
        return vertices
    
    def _polygonize_bezier(self, bezier_data: Dict[str, Any],
                          start_param: float = 0.0, end_param: float = 1.0) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация кривой Безье
        
        Использует рекурсивное разбиение с оценкой отклонения от хорды,
        аналогично алгоритму для NURBS.
        """
        control_points = np.array(bezier_data['control_points'])
        degree = len(control_points) - 1
        
        vertices = []
        self._recursive_bezier_subdivision(
            control_points, start_param, end_param, vertices, 0
        )
        
        return vertices
    
    def _recursive_bezier_subdivision(self, control_points: np.ndarray,
                                     t_start: float, t_end: float, vertices: List[Tuple[float, float]],
                                     depth: int):
        """
        Рекурсивное разбиение кривой Безье
        """
        if depth > self.max_recursion_depth or len(vertices) > self.max_vertices:
            return
        
        # Вычисление точек с использованием алгоритма де Кастельжо
        t_mid = (t_start + t_end) / 2.0
        p_start = self._evaluate_bezier_point(control_points, t_start)
        p_mid = self._evaluate_bezier_point(control_points, t_mid)
        p_end = self._evaluate_bezier_point(control_points, t_end)
        
        # Оценка отклонения
        chord = LineString([p_start, p_end])
        distance = chord.distance(Point(p_mid))
        
        if distance < self.tolerance or (t_end - t_start) < 0.01:
            if not vertices or not np.array_equal(vertices[-1], p_start):
                vertices.append(p_start)
            vertices.append(p_end)
        else:
            # Разбиение контрольных точек
            left_cp, right_cp = self._split_bezier_curve(control_points, 0.5)
            
            self._recursive_bezier_subdivision(
                left_cp, t_start, t_mid, vertices, depth + 1
            )
            self._recursive_bezier_subdivision(
                right_cp, t_mid, t_end, vertices, depth + 1
            )
    
    def _evaluate_bezier_point(self, control_points: np.ndarray, t: float) -> Tuple[float, float]:
        """
        Вычисление точки на кривой Безье с использованием алгоритма де Кастельжо
        """
        n = len(control_points) - 1
        point = np.zeros(2)
        
        for i in range(n + 1):
            binomial = math.comb(n, i)
            term = binomial * (1 - t)**(n - i) * t**i
            point += term * control_points[i]
        
        return tuple(point)
    
    def _split_bezier_curve(self, control_points: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Разбиение кривой Безье на две части в точке t (алгоритм де Кастельжо)
        """
        n = len(control_points) - 1
        left_points = np.zeros((n + 1, 2))
        right_points = np.zeros((n + 1, 2))
        
        # Инициализация
        points = control_points.copy()
        
        left_points[0] = points[0]
        right_points[n] = points[-1]
        
        # Рекурсивное разбиение
        for level in range(1, n + 1):
            new_points = np.zeros((n - level + 1, 2))
            for i in range(n - level + 1):
                new_points[i] = (1 - t) * points[i] + t * points[i + 1]
            
            left_points[level] = new_points[0]
            right_points[n - level] = new_points[-1]
            points = new_points
        
        return left_points, right_points
    
    def _polygonize_spline(self, spline_data: Dict[str, Any],
                          start_param: float = 0.0, end_param: float = 1.0) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация B-сплайна с использованием scipy
        """
        try:
            control_points = np.array(spline_data['control_points'])
            degree = spline_data.get('degree', 3)
            knots = spline_data.get('knots')
            
            if knots is None:
                # Автоматическая генерация узлового вектора
                n = len(control_points)
                knots = np.concatenate((
                    np.zeros(degree),
                    np.linspace(0, 1, n - degree + 1),
                    np.ones(degree)
                ))
            
            # Параметризация для равномерной выборки
            t = np.linspace(start_param, end_param, 100)
            
            # Подготовка данных для scipy
            if len(control_points) <= degree:
                degree = len(control_points) - 1
            
            # Интерполяция координат
            tck_x, tck_y = splprep([control_points[:, 0], control_points[:, 1]], 
                                 k=degree, s=self.tolerance**2, u=np.linspace(0, 1, len(control_points)))[0]
            
            # Вычисление точек на кривой
            spline_points = np.array(splev(t, tck_x)).T
            
            # Адаптивное упрощение через алгоритм Рамера-Дугласа-Пекера
            simplified_points = self._ramer_douglas_peucker(spline_points, self.tolerance)
            
            return [tuple(point) for point in simplified_points]
            
        except ImportError:
            logger.warning("scipy не установлен. Используется упрощенная полигонализация сплайна.")
            return self._simple_spline_approximation(spline_data, start_param, end_param)
        except Exception as e:
            logger.error(f"Ошибка при полигонализации сплайна: {e}")
            raise
    
    def _simple_spline_approximation(self, spline_data: Dict[str, Any],
                                    start_param: float = 0.0, end_param: float = 1.0) -> List[Tuple[float, float]]:
        """
        Упрощенная аппроксимация сплайна линейными сегментами
        """
        control_points = np.array(spline_data['control_points'])
        n_segments = min(20, self.max_vertices // 2)
        
        vertices = []
        for i in range(n_segments + 1):
            t = start_param + (end_param - start_param) * i / n_segments
            
            # Линейная интерполяция между контрольными точками
            idx = int(t * (len(control_points) - 1))
            frac = t * (len(control_points) - 1) - idx
            
            if idx < len(control_points) - 1:
                point = control_points[idx] * (1 - frac) + control_points[idx + 1] * frac
            else:
                point = control_points[-1]
            
            vertices.append(tuple(point))
        
        return vertices
    
    def _ramer_douglas_peucker(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Алгоритм Рамера-Дугласа-Пекера для упрощения полигональной цепи
        
        :param points: Массив точек Nx2
        :param epsilon: Порог расстояния
        :return: Упрощенный массив точек
        """
        if len(points) <= 2:
            return points
        
        # Находим точку с максимальным расстоянием до хорды
        start_point = points[0]
        end_point = points[-1]
        chord = LineString([tuple(start_point), tuple(end_point)])
        
        max_dist = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            dist = chord.distance(Point(points[i]))
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # Если максимальное расстояние больше порога - рекурсивное разбиение
        if max_dist > epsilon:
            left_points = self._ramer_douglas_peucker(points[:max_index+1], epsilon)
            right_points = self._ramer_douglas_peucker(points[max_index:], epsilon)
            
            return np.vstack([
                left_points[:-1],
                right_points
            ])
        else:
            return np.array([start_point, end_point])
    
    def polygonize_complex_shape(self, shape: Union[Polygon, MultiPolygon], 
                                include_holes: bool = True) -> PolygonShape:
        """
        Полигонализация сложной формы (с отверстиями и составными частями)
        
        :param shape: Геометрия Shapely (Polygon или MultiPolygon)
        :param include_holes: Включать ли внутренние контуры (отверстия)
        :return: Объект PolygonShape с адаптивно полигонализированной геометрией
        """
        start_time = time.time()
        
        if isinstance(shape, MultiPolygon):
            # Обработка составной геометрии
            logger.warning("Составная геометрия (MultiPolygon) не полностью поддерживается. Используется первая компонента.")
            shape = list(shape.geoms)[0]
        
        if not isinstance(shape, Polygon):
            raise ValueError("Поддерживаются только Polygon и MultiPolygon")
        
        # Адаптивная полигонализация внешнего контура
        outer_contour = np.array(shape.exterior.coords)[:-1]  # Убираем последнюю точку (дублирует первую)
        simplified_outer = self._ramer_douglas_peucker(outer_contour, self.tolerance)
        
        inner_contours = []
        if include_holes and shape.interiors:
            for interior in shape.interiors:
                inner_contour = np.array(interior.coords)[:-1]
                simplified_inner = self._ramer_douglas_peucker(inner_contour, self.tolerance * 0.8)  # Повышенная точность для отверстий
                inner_contours.append(simplified_inner.tolist())
        
        # Создание объекта PolygonShape
        polygon_shape = PolygonShape(
            simplified_outer.tolist(),
            inner_contours if inner_contours else None,
            name=f"polygonized_shape_{int(time.time())}"
        )
        
        # Расчет статистики
        original_vertices = len(outer_contour) + sum(len(np.array(hole.coords)) for hole in shape.interiors)
        new_vertices = len(simplified_outer) + sum(len(contour) for contour in inner_contours)
        
        if original_vertices > 0:
            reduction_ratio = (original_vertices - new_vertices) / original_vertices
            self.stats['reduction_ratio'] = (self.stats['reduction_ratio'] * (self.stats['total_curves_processed'] - 1) + reduction_ratio) / self.stats['total_curves_processed']
        
        processing_time = time.time() - start_time
        logger.info(f"Полигонализация сложной формы завершена. "
                   f"Вершин: было {original_vertices}, стало {new_vertices} ({reduction_ratio:.1%} сокращение), "
                   f"время: {processing_time:.4f}с")
        
        return polygon_shape
    
    def _postprocess_vertices(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Постобработка списка вершин:
        - Удаление дубликатов
        - Ограничение максимального числа вершин
        - Проверка замкнутости контура
        """
        if not vertices:
            return vertices
        
        # Удаление дубликатов
        unique_vertices = []
        for vertex in vertices:
            if not unique_vertices or not np.allclose(vertex, unique_vertices[-1], atol=1e-6):
                unique_vertices.append(vertex)
        
        # Проверка замкнутости
        if len(unique_vertices) > 2 and np.allclose(unique_vertices[0], unique_vertices[-1], atol=1e-6):
            unique_vertices = unique_vertices[:-1]
        
        # Ограничение максимального числа вершин
        if len(unique_vertices) > self.max_vertices:
            logger.warning(f"Превышено максимальное число вершин ({len(unique_vertices)} > {self.max_vertices}). "
                          f"Производится упрощение.")
            unique_vertices = self._ramer_douglas_peucker(
                np.array(unique_vertices), 
                self.tolerance * 2
            ).tolist()
            
            # Повторная проверка числа вершин
            if len(unique_vertices) > self.max_vertices:
                step = max(1, len(unique_vertices) // self.max_vertices)
                unique_vertices = unique_vertices[::step]
        
        # Замыкание контура, если необходимо
        if unique_vertices and not np.allclose(unique_vertices[0], unique_vertices[-1], atol=1e-6):
            unique_vertices.append(unique_vertices[0])
        
        return unique_vertices
    
    def polygonize_from_dxf_entity(self, entity, layer: str = "") -> Optional[PolygonShape]:
        """
        Полигонализация элемента DXF (адаптация для интеграции с io/dxf_import.py)
        
        :param entity: Объект ezdxf entity
        :param layer: Имя слоя
        :return: Объект PolygonShape или None
        """
        try:
            if entity.dxftype() == 'LWPOLYLINE':
                # Для полилиний используем существующие вершины
                vertices = [(point[0], point[1]) for point in entity.get_points()]
                if entity.is_closed and vertices[0] != vertices[-1]:
                    vertices.append(vertices[0])
                return PolygonShape(vertices, name=f"dxf_{entity.dxftype()}_{layer}")
            
            elif entity.dxftype() == 'CIRCLE':
                center = (entity.dxf.center.x, entity.dxf.center.y)
                radius = entity.dxf.radius
                circle_data = {'center': center, 'radius': radius}
                vertices = self._polygonize_circle(circle_data)
                return PolygonShape(vertices, name=f"dxf_{entity.dxftype()}_{layer}")
            
            elif entity.dxftype() == 'ARC':
                center = (entity.dxf.center.x, entity.dxf.center.y)
                radius = entity.dxf.radius
                start_angle = math.radians(entity.dxf.start_angle)
                end_angle = math.radians(entity.dxf.end_angle)
                circle_data = {'center': center, 'radius': radius}
                vertices = self._polygonize_circle(circle_data, start_angle, end_angle)
                return PolygonShape(vertices, name=f"dxf_{entity.dxftype()}_{layer}")
            
            elif entity.dxftype() == 'ELLIPSE':
                center = (entity.dxf.center.x, entity.dxf.center.y)
                major_axis = entity.dxf.major_axis
                ratio = entity.dxf.ratio
                a = math.hypot(major_axis.x, major_axis.y)
                b = a * ratio
                phi = math.atan2(major_axis.y, major_axis.x)
                ellipse_data = {
                    'center': center,
                    'major_radius': a,
                    'minor_radius': b,
                    'rotation': phi
                }
                vertices = self._polygonize_ellipse(ellipse_data)
                return PolygonShape(vertices, name=f"dxf_{entity.dxftype()}_{layer}")
            
            elif entity.dxftype() == 'SPLINE':
                # Извлечение контрольных точек из SPLINE
                control_points = [(cp.x, cp.y) for cp in entity.control_points]
                if control_points:
                    spline_data = {
                        'control_points': control_points,
                        'degree': entity.dxf.degree
                    }
                    vertices = self._polygonize_spline(spline_data)
                    return PolygonShape(vertices, name=f"dxf_{entity.dxftype()}_{layer}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Ошибка при полигонализации DXF entity {entity.dxftype()} на слое {layer}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики работы полигонализатора
        """
        return {
            'tolerance': self.tolerance,
            'total_curves_processed': self.stats['total_curves_processed'],
            'total_vertices_generated': self.stats['total_vertices_generated'],
            'average_vertices_per_curve': (
                self.stats['total_vertices_generated'] / self.stats['total_curves_processed']
                if self.stats['total_curves_processed'] > 0 else 0
            ),
            'average_processing_time': (
                self.stats['processing_time'] / self.stats['total_curves_processed']
                if self.stats['total_curves_processed'] > 0 else 0
            ),
            'average_reduction_ratio': self.stats['reduction_ratio'],
            'max_vertices_limit': self.max_vertices
        }
    
    def visualize_polygonization(self, original_curve: Any, polygonized_vertices: List[Tuple[float, float]],
                                filename: str = "polygonization_comparison.svg"):
        """
        Визуализация сравнения исходной кривой и ее полигональной аппроксимации
        
        :param original_curve: Исходная кривая для визуализации
        :param polygonized_vertices: Вершины полигональной аппроксимации
        :param filename: Имя файла для сохранения визуализации
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Визуализация исходной кривой (если возможно)
            if hasattr(original_curve, 'exterior'):
                # Для Shapely Polygon
                x, y = original_curve.exterior.xy
                ax.plot(x, y, 'b-', linewidth=2, label='Исходная кривая')
            elif isinstance(original_curve, dict) and 'control_points' in original_curve:
                # Для NURBS/сплайнов - отображение контрольных точек
                cp = np.array(original_curve['control_points'])
                ax.plot(cp[:, 0], cp[:, 1], 'ro-', alpha=0.5, label='Контрольные точки')
            
            # Визуализация полигональной аппроксимации
            if polygonized_vertices:
                poly_x = [v[0] for v in polygonized_vertices]
                poly_y = [v[1] for v in polygonized_vertices]
                ax.plot(poly_x, poly_y, 'g--', linewidth=1.5, label=f'Полигональная аппроксимация ({len(polygonized_vertices)} вершин)')
                
                # Отображение вершин
                ax.plot(poly_x, poly_y, 'go', markersize=4, alpha=0.7)
            
            # Настройка графика
            ax.set_title(f'Адаптивная полигонализация (точность: {self.tolerance} мм)')
            ax.set_xlabel('X (мм)')
            ax.set_ylabel('Y (мм)')
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Сохранение
            plt.savefig(filename, format='svg', bbox_inches='tight')
            plt.close()
            
            logger.info(f"Визуализация полигонализации сохранена в {filename}")
            return True
            
        except ImportError:
            logger.warning("Для визуализации требуется matplotlib. Установите: pip install matplotlib")
            return False
        except Exception as e:
            logger.error(f"Ошибка при визуализации полигонализации: {e}")
            return False
    
    def optimize_for_physics_simulation(self, polygon_shape: PolygonShape, 
                                       max_vertices_physics: int = 50) -> PolygonShape:
        """
        Оптимизация полигональной геометрии для физической симуляции
        
        Снижает детализацию для ускорения расчетов коллизий в динамической системе,
        сохраняя при этом ключевые особенности формы.
        
        :param polygon_shape: Исходная фигура
        :param max_vertices_physics: Максимальное число вершин для физической симуляции
        :return: Оптимизированная фигура
        """
        if len(polygon_shape.outer_contour) <= max_vertices_physics:
            return polygon_shape
        
        # Адаптивное упрощение с сохранением реентрантов (вогнутостей)
        simplified_outer = self._adaptive_simplify_for_physics(
            polygon_shape.outer_contour, max_vertices_physics
        )
        
        simplified_inners = []
        for inner_contour in polygon_shape.inner_contours:
            if len(inner_contour) > max_vertices_physics // 4:
                simplified_inner = self._adaptive_simplify_for_physics(
                    inner_contour, max_vertices_physics // 4
                )
                simplified_inners.append(simplified_inner)
            else:
                simplified_inners.append(inner_contour)
        
        optimized_shape = PolygonShape(
            simplified_outer,
            simplified_inners if simplified_inners else None,
            name=f"{polygon_shape.name}_physics_optimized"
        )
        
        logger.debug(f"Оптимизация для физики: {len(polygon_shape.outer_contour)} -> {len(simplified_outer)} вершин")
        return optimized_shape
    
    def _adaptive_simplify_for_physics(self, contour: List[Tuple[float, float]], 
                                     max_vertices: int) -> List[Tuple[float, float]]:
        """
        Адаптивное упрощение контура с сохранением реентрантов для физической симуляции
        """
        if len(contour) <= max_vertices:
            return contour
        
        # Расчет кривизны в каждой точке
        curvatures = self._calculate_curvatures(contour)
        
        # Выбор ключевых точек на основе кривизны
        key_indices = np.argsort(curvatures)[::-1][:max_vertices//2]
        
        # Добавление равномерно распределенных точек
        step = max(1, len(contour) // (max_vertices - len(key_indices)))
        uniform_indices = np.arange(0, len(contour), step)
        
        # Объединение и сортировка индексов
        all_indices = np.unique(np.concatenate([key_indices, uniform_indices]))
        all_indices = np.sort(all_indices[:max_vertices])
        
        # Формирование упрощенного контура
        simplified_contour = [contour[i] for i in all_indices]
        
        return simplified_contour
    
    def _calculate_curvatures(self, contour: List[Tuple[float, float]]) -> np.ndarray:
        """
        Расчет кривизны контура в каждой точке
        """
        n = len(contour)
        curvatures = np.zeros(n)
        
        for i in range(n):
            p_prev = np.array(contour[(i - 1) % n])
            p_curr = np.array(contour[i])
            p_next = np.array(contour[(i + 1) % n])
            
            # Векторы
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # Нормализация
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                curvatures[i] = 0
                continue
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Кривизна как угол между векторами
            dot_product = np.dot(v1, v2)
            dot_product = max(min(dot_product, 1.0), -1.0)  # Ограничение для арккосинуса
            angle = math.acos(dot_product)
            
            # Знак кривизны зависит от ориентации
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            curvature = angle * (1 if cross_product > 0 else -1)
            
            curvatures[i] = abs(curvature)
        
        return curvatures
    
    def __str__(self):
        """Строковое представление для отладки"""
        stats = self.get_statistics()
        return (f"AdaptivePolygonizer(tolerance={self.tolerance}mm, "
                f"max_vertices={self.max_vertices}, "
                f"processed_curves={stats['total_curves_processed']}, "
                f"avg_vertices={stats['average_vertices_per_curve']:.1f})")