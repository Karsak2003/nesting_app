
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from core.geometry import PolygonShape
from shapely.geometry import Polygon, Point, LinearRing, MultiPolygon
from shapely.ops import unary_union
import time
import math
import logging
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class SignedDistanceField:
    """
    Signed Distance Field (SDF) для геометрических объектов
    
    Реализует функционал, описанный в разделах 2.2.3 и 2.3.4 диссертации:
    - Вычисление расстояния от точки до границ фигуры с учетом знака
    - Билинейная интерполяция для повышения точности
    - Учет многосвязных фигур с отверстиями
    - Градиент поля для расчета сил взаимодействия
    
    Согласно разделу 2.2.5, при разрешении сетки 0.1 мм погрешность не превышает 0.07 мм,
    что соответствует требованиям авиастроения.
    """
    
    def __init__(self, polygon: Polygon, bounds: Tuple[float, float, float, float], 
                 resolution: float = 0.5, padding: float = 5.0, use_adaptive: bool = True):
        """
        Инициализация SDF для заданного полигона
        
        :param polygon: Полигон (может содержать отверстия)
        :param bounds: Границы области (min_x, min_y, max_x, max_y)
        :param resolution: Разрешение сетки в мм (точность)
        :param padding: Отступ от границ фигуры для расширения области
        :param use_adaptive: Использовать адаптивное разрешение (повышенная точность около границ)
        """
        self.polygon = polygon
        self.resolution = resolution
        self.use_adaptive = use_adaptive
        self.padding = padding
        self.bounds = self._expand_bounds(bounds, padding)
        
        # Параметры сетки
        self.min_x, self.min_y, self.max_x, self.max_y = self.bounds
        self.width = int((self.max_x - self.min_x) / self.resolution) + 1
        self.height = int((self.max_y - self.min_y) / self.resolution) + 1
        
        # Инициализация массивов расстояний
        self.distance_field = np.zeros((self.height, self.width))
        self.gradient_field = np.zeros((self.height, self.width, 2))  # x, y компоненты градиента
        
        # Дерево для ускорения поиска ближайших точек
        self.kdtree = None
        self.boundary_points = None
        
        # Время построения
        self.build_time = 0.0
        
        # Построение поля
        self._build_field()
    
    def _expand_bounds(self, bounds: Tuple[float, float, float, float], padding: float) -> Tuple[float, float, float, float]:
        """
        Расширение границ для учета отступа
        
        :param bounds: Исходные границы (min_x, min_y, max_x, max_y)
        :param padding: Отступ в мм
        :return: Расширенные границы
        """
        min_x, min_y, max_x, max_y = bounds
        return (
            min_x - padding,
            min_y - padding,
            max_x + padding,
            max_y + padding
        )
    
    def _build_field(self):
        """
        Построение Signed Distance Field
        
        Алгоритм:
        1. Дискретизация границ полигона
        2. Построение начального поля расстояний (distance transform)
        3. Установка знака в зависимости от принадлежности области
        4. Вычисление градиента поля
        5. (Опционально) адаптивное сглаживание вблизи границ
        """
        start_time = time.time()
        logger.debug(f"Построение SDF для полигона с разрешением {self.resolution} мм...")
        
        # 1. Дискретизация границ полигона
        self.boundary_points = self._discretize_boundary(self.polygon, max_segment_length=self.resolution)
        
        # 2. Создание бинарного изображения фигуры
        binary_image = self._create_binary_image()
        
        # 3. Построение поля расстояний с помощью distance transform
        if self.use_adaptive:
            # Адаптивный метод с повышенной точностью вблизи границ
            self.distance_field = self._adaptive_distance_transform(binary_image)
        else:
            # Стандартный метод
            self.distance_field = self._distance_transform(binary_image)
        
        # 4. Вычисление градиента поля
        self._compute_gradient_field()
        
        # 5. Построение KD-дерева для ускорения поиска
        self._build_kdtree()
        
        self.build_time = time.time() - start_time
        logger.debug(f"SDF построено за {self.build_time:.4f} секунд. "
                    f"Размер сетки: {self.width}x{self.height}")
    
    def _discretize_boundary(self, polygon: Polygon, max_segment_length: float = 1.0) -> np.ndarray:
        """
        Дискретизация границ полигона на точки
        
        :param polygon: Полигон с возможными отверстиями
        :param max_segment_length: Максимальная длина сегмента в мм
        :return: Массив точек границ [(x1, y1), (x2, y2), ...]
        """
        boundary_points = []
        
        # Внешняя граница
        outer_boundary = list(polygon.exterior.coords)[:-1]  # Убираем последнюю точку (дублирует первую)
        boundary_points.extend(self._subdivide_segment(outer_boundary, max_segment_length))
        
        # Внутренние границы (отверстия)
        for interior in polygon.interiors:
            interior_boundary = list(interior.coords)[:-1]
            boundary_points.extend(self._subdivide_segment(interior_boundary, max_segment_length))
        
        return np.array(boundary_points)
    
    def _subdivide_segment(self, segment_points: List[Tuple[float, float]], max_length: float) -> List[Tuple[float, float]]:
        """
        Разбиение сегмента на части заданной максимальной длины
        
        :param segment_points: Точки сегмента
        :param max_length: Максимальная длина подсегмента
        :return: Список точек с добавленными промежуточными точками
        """
        result = [segment_points[0]]
        
        for i in range(1, len(segment_points)):
            p1 = np.array(segment_points[i-1])
            p2 = np.array(segment_points[i])
            segment_length = np.linalg.norm(p2 - p1)
            
            if segment_length > max_length:
                # Число промежуточных точек
                num_points = int(np.ceil(segment_length / max_length))
                for j in range(1, num_points):
                    t = j / num_points
                    new_point = p1 + t * (p2 - p1)
                    result.append(tuple(new_point))
            result.append(tuple(p2))
        
        return result
    
    def _create_binary_image(self) -> np.ndarray:
        """
        Создание бинарного изображения фигуры на сетке
        
        :return: Бинарное изображение (1 - внутри фигуры, 0 - снаружи)
        """
        binary_image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Заполнение фигуры
        for i in range(self.height):
            y = self.min_y + i * self.resolution
            for j in range(self.width):
                x = self.min_x + j * self.resolution
                point = Point(x, y)
                if self.polygon.contains(point):
                    binary_image[i, j] = 1
        
        # Морфологическое сглаживание для устранения артефактов
        binary_image = ndimage.binary_dilation(binary_image, iterations=1)
        binary_image = ndimage.binary_erosion(binary_image, iterations=1)
        
        return binary_image
    
    def _distance_transform(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Стандартное преобразование расстояний
        
        :param binary_image: Бинарное изображение фигуры
        :return: Поле расстояний с правильными знаками
        """
        # Расстояние до границы для точек вне фигуры
        dist_outside = ndimage.distance_transform_edt(1 - binary_image)
        
        # Расстояние до границы для точек внутри фигуры
        dist_inside = ndimage.distance_transform_edt(binary_image)
        
        # Комбинирование с учетом знака
        distance_field = np.where(binary_image == 1, -dist_inside, dist_outside)
        
        # Преобразование в миллиметры
        distance_field *= self.resolution
        
        return distance_field
    
    def _adaptive_distance_transform(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Адаптивное преобразование расстояний с повышенной точностью вблизи границ
        
        Согласно разделу 2.2.5, адаптивная полигонализация с точностью 0.1 мм
        обеспечивает погрешность менее 0.07 мм при сохранении вычислительной эффективности.
        """
        # Стандартное преобразование расстояний
        distance_field = self._distance_transform(binary_image)
        
        # Повышенная точность вблизи границ (в зоне ±2*resolution)
        boundary_mask = np.abs(distance_field) <= 2 * self.resolution
        
        if np.any(boundary_mask):
            # Для точек вблизи границы используем точный расчет расстояния до ближайшей точки границы
            coords = np.where(boundary_mask)
            for i, j in zip(coords[0], coords[1]):
                x = self.min_x + j * self.resolution
                y = self.min_y + i * self.resolution
                point = np.array([x, y])
                
                # Точное расстояние до ближайшей точки границы
                distances = np.linalg.norm(self.boundary_points - point, axis=1)
                min_distance = np.min(distances)
                
                # Определение знака (внутри/снаружи фигуры)
                if binary_image[i, j] == 1:
                    min_distance = -min_distance
                
                distance_field[i, j] = min_distance
        
        return distance_field
    
    def _compute_gradient_field(self):
        """
        Вычисление градиента SDF
        
        Градиент SDF указывает направление наибольшего увеличения расстояния
        и используется для расчета сил отталкивания (раздел 2.3.4).
        """
        # Частные производные по x и y
        dx = ndimage.sobel(self.distance_field, axis=1, mode='constant') / (8.0 * self.resolution)
        dy = ndimage.sobel(self.distance_field, axis=0, mode='constant') / (8.0 * self.resolution)
        
        # Нормализация градиента
        magnitude = np.sqrt(dx**2 + dy**2)
        mask = magnitude > 1e-6  # Избегаем деления на ноль
        
        self.gradient_field[mask, 0] = dx[mask] / magnitude[mask]
        self.gradient_field[mask, 1] = dy[mask] / magnitude[mask]
        
        # Для точек с нулевым градиентом используем направление к центру масс
        if not np.all(mask):
            centroid = np.array(self.polygon.centroid.coords[0])
            zero_grad_indices = np.where(~mask)
            
            for i, j in zip(zero_grad_indices[0], zero_grad_indices[1]):
                x = self.min_x + j * self.resolution
                y = self.min_y + i * self.resolution
                direction = np.array([x, y]) - centroid
                norm = np.linalg.norm(direction)
                
                if norm > 1e-6:
                    self.gradient_field[i, j, 0] = direction[0] / norm
                    self.gradient_field[i, j, 1] = direction[1] / norm
    
    def _build_kdtree(self):
        """
        Построение KD-дерева для ускорения поиска ближайших точек
        
        Это обеспечивает сложность O(log n) для запросов расстояния,
        что критично для масштабируемости системы (раздел 2.2.5).
        """
        self.kdtree = cKDTree(self.boundary_points)
    
    def evaluate(self, point: Tuple[float, float], compute_gradient: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        Вычисление SDF для заданной точки
        
        :param point: Точка (x, y)
        :param compute_gradient: Вычислять ли градиент
        :return: (distance, gradient) - расстояние и градиент (если запрошен)
        """
        x, y = point
        
        # Проверка выхода за границы области
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            # Внешнее расстояние вычисляем напрямую
            if self.kdtree is not None:
                distance, idx = self.kdtree.query([x, y])
                # Определение знака через принадлежность полигону
                if self.polygon.contains(Point(x, y)):
                    distance = -distance
                gradient = self._compute_boundary_gradient(x, y, idx) if compute_gradient else None
                return distance, gradient
            else:
                return float('inf'), None
        
        # Вычисление координат в сетке
        j = int((x - self.min_x) / self.resolution)
        i = int((y - self.min_y) / self.resolution)
        
        # Проверка корректности индексов
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return float('inf'), None
        
        # Билинейная интерполяция
        x_frac = (x - self.min_x) / self.resolution - j
        y_frac = (y - self.min_y) / self.resolution - i
        
        # Значения расстояния в четырех соседних узлах
        d00 = self.distance_field[i, j]
        d10 = self.distance_field[i, min(j+1, self.width-1)]
        d01 = self.distance_field[min(i+1, self.height-1), j]
        d11 = self.distance_field[min(i+1, self.height-1), min(j+1, self.width-1)]
        
        # Билинейная интерполяция
        distance = (
            d00 * (1 - x_frac) * (1 - y_frac) +
            d10 * x_frac * (1 - y_frac) +
            d01 * (1 - x_frac) * y_frac +
            d11 * x_frac * y_frac
        )
        
        # Вычисление градиента при необходимости
        gradient = None
        if compute_gradient:
            # Значения градиента в четырех соседних узлах
            g00 = self.gradient_field[i, j]
            g10 = self.gradient_field[i, min(j+1, self.width-1)]
            g01 = self.gradient_field[min(i+1, self.height-1), j]
            g11 = self.gradient_field[min(i+1, self.height-1), min(j+1, self.width-1)]
            
            # Билинейная интерполяция градиента
            grad_x = (
                g00[0] * (1 - x_frac) * (1 - y_frac) +
                g10[0] * x_frac * (1 - y_frac) +
                g01[0] * (1 - x_frac) * y_frac +
                g11[0] * x_frac * y_frac
            )
            
            grad_y = (
                g00[1] * (1 - x_frac) * (1 - y_frac) +
                g10[1] * x_frac * (1 - y_frac) +
                g01[1] * (1 - x_frac) * y_frac +
                g11[1] * x_frac * y_frac
            )
            
            gradient = np.array([grad_x, grad_y])
        
        return distance, gradient
    
    def _compute_boundary_gradient(self, x: float, y: float, boundary_idx: int) -> np.ndarray:
        """
        Вычисление градиента на границе для точек вне основной области
        
        :param x, y: Координаты точки
        :param boundary_idx: Индекс ближайшей точки границы
        :return: Градиент
        """
        boundary_point = self.boundary_points[boundary_idx]
        
        # Направление от границы к точке
        direction = np.array([x, y]) - boundary_point
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # Если точка на границе, используем нормаль к границе
            # Находим соседние точки для аппроксимации касательной
            idx_prev = (boundary_idx - 1) % len(self.boundary_points)
            idx_next = (boundary_idx + 1) % len(self.boundary_points)
            
            prev_point = self.boundary_points[idx_prev]
            next_point = self.boundary_points[idx_next]
            
            # Касательный вектор
            tangent = next_point - prev_point
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm < 1e-6:
                return np.array([1.0, 0.0])  # Нормаль по умолчанию
            
            # Нормаль к границе (поворот касательной на 90 градусов)
            tangent = tangent / tangent_norm
            normal = np.array([-tangent[1], tangent[0]])
            
            # Определение направления нормали (внутрь или наружу)
            test_point = boundary_point + 0.1 * normal
            if self.polygon.contains(Point(test_point[0], test_point[1])):
                return normal
            else:
                return -normal
        else:
            # Направление от границы к точке
            return direction / distance
    
    def visualize(self, filename: str = None, show: bool = True):
        """
        Визуализация Signed Distance Field
        
        :param filename: Имя файла для сохранения (если None, не сохраняется)
        :param show: Показывать ли изображение
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Создание кастомной цветовой карты
        colors = [
            (0.0, 0.0, 1.0),    # Синий для отрицательных значений
            (0.5, 0.5, 1.0),    # Светло-синий
            (1.0, 1.0, 1.0),    # Белый для нуля
            (1.0, 0.5, 0.5),    # Светло-красный
            (1.0, 0.0, 0.0)     # Красный для положительных значений
        ]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Отрисовка поля расстояний
        extent = [self.min_x, self.max_x, self.min_y, self.max_y]
        im = ax.imshow(
            self.distance_field, 
            extent=extent,
            origin='lower',
            cmap=cmap,
            vmin=-np.max(np.abs(self.distance_field)),
            vmax=np.max(np.abs(self.distance_field))
        )
        
        # Добавление контура фигуры
        x_outer, y_outer = self.polygon.exterior.xy
        ax.plot(x_outer, y_outer, 'k-', linewidth=2, label='Граница фигуры')
        
        for interior in self.polygon.interiors:
            x_inner, y_inner = interior.xy
            ax.plot(x_inner, y_inner, 'k-', linewidth=2)
        
        # Добавление цветовой шкалы
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Signed Distance (мм)')
        
        # Настройка графика
        ax.set_title(f'Signed Distance Field (разрешение: {self.resolution} мм)')
        ax.set_xlabel('X (мм)')
        ax.set_ylabel('Y (мм)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавление изолиний
        contour_levels = np.linspace(
            -np.max(np.abs(self.distance_field)), 
            np.max(np.abs(self.distance_field)), 
            11
        )
        ax.contour(
            self.distance_field, 
            levels=contour_levels,
            extent=extent,
            colors='k',
            alpha=0.3,
            linewidths=0.5
        )
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"SDF визуализировано и сохранено в {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_field_stats(self) -> Dict[str, float]:
        """
        Получение статистики о поле расстояний
        
        :return: Словарь со статистикой
        """
        return {
            'min_distance': np.min(self.distance_field),
            'max_distance': np.max(self.distance_field),
            'mean_distance': np.mean(self.distance_field),
            'std_distance': np.std(self.distance_field),
            'resolution': self.resolution,
            'grid_width': self.width,
            'grid_height': self.height,
            'build_time': self.build_time
        }
    
    def __str__(self):
        """Строковое представление для отладки"""
        stats = self.get_field_stats()
        return (f"SignedDistanceField(resolution={self.resolution}mm, "
                f"size={self.width}x{self.height}, "
                f"min={stats['min_distance']:.2f}mm, max={stats['max_distance']:.2f}mm)")

def compute_sdf_field(polygon: Polygon, resolution: float = 0.5, 
                     bounds: Optional[Tuple[float, float, float, float]] = None,
                     padding: float = 5.0) -> SignedDistanceField:
    """
    Создание Signed Distance Field для заданного полигона
    
    :param polygon: Полигон для создания SDF
    :param resolution: Разрешение сетки в мм
    :param bounds: Границы области (если None, используются границы полигона)
    :param padding: Отступ от границ фигуры
    :return: Объект SignedDistanceField
    """
    # Определение границ, если не заданы
    if bounds is None:
        min_x, min_y, max_x, max_y = polygon.bounds
        bounds = (min_x, min_y, max_x, max_y)
    
    # Создание SDF
    sdf = SignedDistanceField(
        polygon=polygon,
        bounds=bounds,
        resolution=resolution,
        padding=padding
    )
    
    return sdf

def create_sdf_for_shape(shape: PolygonShape, resolution: float = 0.5, 
                        sheet_size: Optional[Tuple[float, float]] = None) -> SignedDistanceField:
    """
    Создание SDF для фигуры из системы раскроя
    
    :param shape: Фигура из системы раскроя
    :param resolution: Разрешение сетки в мм
    :param sheet_size: Размеры листа для определения границ
    :return: Объект SignedDistanceField
    """
    # Получение полигона из фигуры
    polygon = shape.polygon
    
    # Определение границ
    if sheet_size is not None:
        bounds = (0, 0, sheet_size[0], sheet_size[1])
    else:
        min_x, min_y, max_x, max_y = polygon.bounds
        bounds = (min_x, min_y, max_x, max_y)
    
    # Создание SDF
    sdf = compute_sdf_field(
        polygon=polygon,
        resolution=resolution,
        bounds=bounds,
        padding=5.0
    )
    
    return sdf

def batch_create_sdf(shapes: List[PolygonShape], resolution: float = 0.5,
                    sheet_size: Tuple[float, float] = (2000.0, 1000.0),
                    parallel: bool = False) -> List[SignedDistanceField]:
    """
    Пакетное создание SDF для множества фигур
    
    :param shapes: Список фигур
    :param resolution: Разрешение сетки в мм
    :param sheet_size: Размеры листа
    :param parallel: Использовать ли параллельные вычисления
    :return: Список SDF полей
    """
    start_time = time.time()
    logger.info(f"Пакетное создание SDF для {len(shapes)} фигур...")
    
    if parallel:
        # Попытка использования параллельных вычислений
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            # Функция для создания SDF для одной фигуры
            def create_sdf_for_single_shape(shape_idx):
                shape = shapes[shape_idx]
                return create_sdf_for_shape(shape, resolution, sheet_size)
            
            # Параллельное выполнение
            sdf_fields = []
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(create_sdf_for_single_shape, i) for i in range(len(shapes))]
                for future in as_completed(futures):
                    sdf_fields.append(future.result())
            
            logger.info(f"Параллельное создание SDF завершено за {time.time() - start_time:.2f} секунд")
            return sdf_fields
            
        except ImportError as e:
            logger.warning(f"Не удалось импортировать concurrent.futures: {e}")
            logger.warning("Переключение на последовательный режим...")
    
    # Последовательное создание
    sdf_fields = []
    for i, shape in enumerate(shapes):
        if i % 10 == 0:
            logger.debug(f"Создание SDF для фигуры {i+1}/{len(shapes)}...")
        
        sdf = create_sdf_for_shape(shape, resolution, sheet_size)
        sdf_fields.append(sdf)
    
    total_time = time.time() - start_time
    logger.info(f"Последовательное создание SDF завершено за {total_time:.2f} секунд")
    logger.info(f"Среднее время на фигуру: {total_time/len(shapes):.4f} секунд")
    
    return sdf_fields