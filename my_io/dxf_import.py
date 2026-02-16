import traceback

import ezdxf
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from core.geometry import PolygonShape
import math
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union
import warnings
import time
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

class DXFImporter:
    """
    Класс для импорта геометрических данных из DXF файлов
    
    Реализует функционал, описанный в главах 2 и 3:
    - Адаптивная полигонализация NURBS и криволинейных сегментов
    - Поддержка фигур с отверстиями и составных геометрий
    - Извлечение технологических ограничений из слоев и атрибутов
    - Соответствие промышленным стандартам (ГОСТ Р ИСО 10303-21)
    """
    
    def __init__(self, tolerance: float = 0.1, max_vertices: int = 200):
        """
        Инициализация DXF импортера
        
        :param tolerance: Точность аппроксимации кривых в мм (согласно разделу 2.1.7)
        :param max_vertices: Максимальное число вершин для одной фигуры
        """
        self.tolerance = tolerance
        self.max_vertices = max_vertices
        self.layer_mapping = {}  # Сопоставление слоев и типов геометрии
        self.attributes = {}    # Метаданные из DXF
        
    def import_file(self, filename: str, sheet_size: Optional[Tuple[float, float]] = None) -> List[PolygonShape]:
        """
        Импорт фигур из DXF файла
        
        :param filename: Путь к DXF файлу
        :param sheet_size: Размеры листа (ширина, высота) для нормализации координат
        :return: Список импортированных фигур
        """
        start_time = time.time()
        logger.info(f"Начало импорта DXF файла: {filename}")
        
        try:
            # Загрузка DXF документа
            doc = ezdxf.readfile(filename)
            msp = doc.modelspace()
            logger.info(f"DXF файл загружен. Версия: {doc.dxfversion}")
            
            # Анализ структуры документа
            self._analyze_document_structure(doc)
            
            # Извлечение фигур из пространства моделей
            shapes = self._extract_shapes_from_modelspace(msp, doc, sheet_size)
            
            # Обработка составных фигур и отверстий
            shapes = self._process_composite_geometries(shapes)
            
            # Извлечение технологических ограничений
            self._extract_technological_constraints(doc, shapes)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Импорт завершен успешно. Загружено {len(shapes)} фигур за {elapsed_time:.2f} секунд")
            
            return shapes
            
        except IOError as e:
            logger.error(f"Ошибка чтения файла {filename}: {e}")
            raise
        except ezdxf.DXFStructureError as e:
            logger.error(f"Ошибка структуры DXF файла {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при импорте {filename}: {e}")
            raise
    
    def _analyze_document_structure(self, doc: ezdxf.document.Drawing):
        """
        Анализ структуры DXF документа для определения семантики слоев
        
        Согласно разделу 2.5.6, интеграция с промышленными стандартами требует
        правильной интерпретации слоев и атрибутов DXF файла.
        """
        logger.info("Анализ структуры DXF документа...")
        
        # Стандартные имена слоев для различных типов объектов
        standard_layers = {
            'PARTS': 'figures',
            'FIGURES': 'figures',
            'DETAILS': 'figures',
            'DEFECTS': 'defects',
            'HOLES': 'holes',
            'FORBIDDEN': 'forbidden_areas',
            'CONSTRAINTS': 'constraints',
            'BOUNDARY': 'sheet_boundary',
            'SHEET': 'sheet_boundary'
        }
        
        # Анализ существующих слоев в документе
        self.layer_mapping = {}
        for layer in doc.layers:
            layer_name = layer.dxf.name.upper()
            
            # Поиск соответствия в стандартных слоях
            for std_name, semantic_type in standard_layers.items():
                if std_name in layer_name:
                    self.layer_mapping[layer.dxf.name] = semantic_type
                    logger.debug(f"Слой '{layer.dxf.name}' отнесен к типу '{semantic_type}'")
                    break
            else:
                # Если нет явного соответствия, используем эвристики
                if 'DEFECT' in layer_name or 'FLAW' in layer_name:
                    self.layer_mapping[layer.dxf.name] = 'defects'
                elif 'HOLE' in layer_name or 'CUTOUT' in layer_name:
                    self.layer_mapping[layer.dxf.name] = 'holes'
                elif 'BOUNDARY' in layer_name or 'SHEET' in layer_name:
                    self.layer_mapping[layer.dxf.name] = 'sheet_boundary'
                else:
                    self.layer_mapping[layer.dxf.name] = 'figures'
        
        # Извлечение атрибутов документа
        self.attributes = {}
        try:
            for entity in doc.modelspace().query('INSERT'):
                for attrib in entity.attribs:
                    self.attributes[attrib.dxf.tag] = attrib.dxf.text
        except Exception as e:
            logger.warning(f"Не удалось извлечь атрибуты: {e}")
    
    def _extract_shapes_from_modelspace(self, msp: ezdxf.layouts.Modelspace, 
                                       doc: ezdxf.document.Drawing,
                                       sheet_size: Optional[Tuple[float, float]] = None) -> List[PolygonShape]:
        """
        Извлечение фигур из пространства моделей
        
        Обрабатывает различные типы геометрии в соответствии с разделом 2.1:
        - Полилинии (LWPOLYLINE)
        - Окружности и дуги (CIRCLE, ARC)
        - Эллипсы (ELLIPSE)
        - Сплайны (SPLINE)
        """
        logger.info("Извлечение фигур из пространства моделей...")
        shapes = []
        shape_counter = 1
        
        # Сбор всех геометрических примитивов по типам
        geometry_by_layer = {}
        unprocessed_types = set()
        
        for entity in msp:
            layer_name = entity.dxf.layer
            if layer_name not in geometry_by_layer:
                geometry_by_layer[layer_name] = []
            
            geometry_by_layer[layer_name].append(entity)
            unprocessed_types.add(entity.dxftype())
        
        # Обработка геометрии по слоям
        for layer_name, entities in geometry_by_layer.items():
            semantic_type = self.layer_mapping.get(layer_name, 'figures')
            
            if semantic_type in ['sheet_boundary', 'forbidden_areas', 'defects', 'holes']:
                logger.info(f"Пропуск служебного слоя '{layer_name}' (тип: {semantic_type}) - обрабатывается отдельно")
                continue  # Эти типы обрабатываются отдельно
            
            logger.info(f"Обработка слоя '{layer_name}' (тип: {semantic_type}) с {len(entities)} объектов")
            
            processed_types = set()
            skipped_types = set()
            
            for entity in entities:
                entity_type = entity.dxftype()
                processed_types.add(entity_type)
                
                try:
                    if entity_type == 'LWPOLYLINE':
                        contour = self._extract_lwpolyline(entity)
                        if contour and len(contour) >= 3:  # Минимум 3 точки для полигона
                            shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                            shapes.append(shape)
                            shape_counter += 1
                    
                    elif entity_type == 'POLYLINE':
                        contour = self._extract_polyline(entity)
                        if contour and len(contour) >= 3:
                            shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                            shapes.append(shape)
                            shape_counter += 1
                    
                    elif entity_type == 'CIRCLE':
                        contour = self._extract_circle(entity)
                        shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                        shapes.append(shape)
                        shape_counter += 1
                    
                    elif entity_type == 'ARC':
                        contour = self._extract_arc(entity)
                        if contour and len(contour) >= 3:
                            shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                            shapes.append(shape)
                            shape_counter += 1
                    
                    elif entity_type == 'ELLIPSE':
                        contour = self._extract_ellipse(entity)
                        shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                        shapes.append(shape)
                        shape_counter += 1
                    
                    elif entity_type == 'SPLINE':
                        contour = self._extract_spline(entity)
                        if contour and len(contour) >= 3:
                            shape = self._create_shape_from_contour(contour, f"shape_{shape_counter}", layer_name)
                            shapes.append(shape)
                            shape_counter += 1
                    
                    elif entity_type == 'INSERT':
                        # Обработка блоков (возможно, составные фигуры)
                        block_shapes = self._extract_block(entity, doc)
                        shapes.extend(block_shapes)
                        shape_counter += len(block_shapes)
                    
                    else:
                        skipped_types.add(entity_type)
                
                except ValueError as e:
                    logger.debug(f"Пропуск вырожденной фигуры {entity_type} на слое {layer_name}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"{traceback.format_exc()}")
                    logger.warning(f"Ошибка при обработке объекта {entity_type} на слое {layer_name}: {e}")
                    continue
            
            if skipped_types:
                logger.warning(f"На слое '{layer_name}' обнаружены неподдерживаемые типы объектов: {', '.join(sorted(skipped_types))}")
        
        all_skipped = unprocessed_types - processed_types
        if all_skipped:
            logger.warning(f"В файле обнаружены типы объектов, которые не были обработаны: {', '.join(sorted(all_skipped))}")
        
        logger.info(f"Извлечено {len(shapes)} базовых фигур")
        return shapes
    
    def _extract_lwpolyline(self, polyline) -> List[Tuple[float, float]]:
        """
        Извлечение контура из легкой полилинии (LWPOLYLINE)
        
        Согласно разделу 2.1.2, полигональное представление является основой
        для большинства вычислений в системе.
        """
        vertices = []
        for point in polyline.get_points():
            x, y = point[0], point[1]
            vertices.append((x, y))
        
        # Замыкание контура, если необходимо
        if polyline.is_closed and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        
        return vertices
    
    def _extract_polyline(self, polyline) -> List[Tuple[float, float]]:
        """
        Извлечение контура из старой полилинии (POLYLINE)
        
        POLYLINE - это старый формат DXF, который использует отдельные VERTEX объекты.
        """
        vertices = []
        
        is_closed = polyline.is_closed
        
        for vertex in polyline.vertices:
            if vertex.dxftype() == 'VERTEX':
                x = vertex.dxf.location.x
                y = vertex.dxf.location.y
                vertices.append((x, y))
        
        if is_closed and len(vertices) > 0 and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        
        return vertices
    
    def _extract_circle(self, circle) -> List[Tuple[float, float]]:
        """
        Извлечение контура окружности с адаптивной полигонализацией
        
        Реализует методы, описанные в разделе 2.1.7 для обработки
        криволинейных сегментов с заданной точностью.
        """
        center = (circle.dxf.center.x, circle.dxf.center.y)
        radius = circle.dxf.radius
        
        # Расчет количества сегментов для достижения заданной точности
        # Формула: n = ceil(pi / acos(1 - tolerance/radius))
        if radius < self.tolerance:
            return [center, center]  # Вырожденный случай
        
        angle_step = 2 * math.acos(1 - self.tolerance / radius)
        num_segments = max(8, int(2 * math.pi / angle_step))  # Минимум 8 сегментов
        
        vertices = []
        for i in range(num_segments):
            angle = 2 * math.pi * i / num_segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        
        vertices.append(vertices[0])  # Замыкание контура
        return vertices
    
    def _extract_arc(self, arc) -> List[Tuple[float, float]]:
        """
        Извлечение контура дуги с адаптивной полигонализацией
        """
        center = (arc.dxf.center.x, arc.dxf.center.y)
        radius = arc.dxf.radius
        start_angle = math.radians(arc.dxf.start_angle)
        end_angle = math.radians(arc.dxf.end_angle)
        
        # Нормализация углов
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        
        angle_range = end_angle - start_angle
        
        # Расчет количества сегментов
        if radius < self.tolerance:
            return []
        
        angle_step = 2 * math.acos(1 - self.tolerance / radius)
        num_segments = max(4, int(angle_range / angle_step))
        
        vertices = []
        for i in range(num_segments + 1):
            angle = start_angle + angle_range * i / num_segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        
        return vertices
    
    def _extract_ellipse(self, ellipse) -> List[Tuple[float, float]]:
        """
        Извлечение контура эллипса с адаптивной полигонализацией
        
        Эллипс параметрически представляется как:
        x = cx + a*cos(t)*cos(phi) - b*sin(t)*sin(phi)
        y = cy + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)
        где a, b - полуоси, phi - угол поворота
        """
        center = (ellipse.dxf.center.x, ellipse.dxf.center.y)
        major_axis = ellipse.dxf.major_axis
        ratio = ellipse.dxf.ratio
        start_param = ellipse.dxf.start_param
        end_param = ellipse.dxf.end_param
        
        # Вычисление параметров эллипса
        a = math.hypot(major_axis.x, major_axis.y)  # Большая полуось
        b = a * ratio  # Малая полуось
        phi = math.atan2(major_axis.y, major_axis.x)  # Угол поворота
        
        # Нормализация параметров
        if end_param < start_param:
            end_param += 2 * math.pi
        
        param_range = end_param - start_param
        
        # Расчет количества сегментов для достижения точности
        max_curvature = max(1/a, 1/b)  # Максимальная кривизна
        angle_step = 2 * math.acos(1 - self.tolerance * max_curvature)
        num_segments = max(8, int(param_range / angle_step))
        
        vertices = []
        for i in range(num_segments + 1):
            t = start_param + param_range * i / num_segments
            x = (center[0] + a * math.cos(t) * math.cos(phi) - 
                 b * math.sin(t) * math.sin(phi))
            y = (center[1] + a * math.cos(t) * math.sin(phi) + 
                 b * math.sin(t) * math.cos(phi))
            vertices.append((x, y))
        
        if abs(param_range - 2 * math.pi) < 1e-6:  # Замкнутый эллипс
            vertices.append(vertices[0])
        
        return vertices
    
    def _extract_spline(self, spline) -> List[Tuple[float, float]]:
        """
        Извлечение контура сплайна с адаптивной полигонализацией
        
        Для NURBS-кривых используется подход из раздела 2.1.7,
        обеспечивающий точность аппроксимации ≤ 0.1 мм.
        """
        # Проверка, является ли сплайн замкнутым
        is_closed = spline.closed
        
        # Извлечение контрольных точек и узлов
        control_points = [(pt.x, pt.y) for pt in spline.control_points]
        knots = spline.knots if spline.knots else None
        
        # Адаптивная полигонализация сплайна
        vertices = self._adaptive_spline_polyline(spline, is_closed)
        
        return vertices
    
    def _adaptive_spline_polyline(self, spline, is_closed: bool) -> List[Tuple[float, float]]:
        """
        Адаптивная полигонализация сплайна с рекурсивным разбиением
        
        Реализует алгоритм из раздела 2.1.7, обеспечивающий
        оптимальное соотношение точности и числа вершин.
        """
        def evaluate_spline(t):
            """Вычисление точки на сплайне для параметра t"""
            bspline = spline.construction_tool()
            point = bspline.point(t)
            return point.x, point.y
        
        def recursive_subdivision(t_start, t_end, points):
            """Рекурсивное разбиение сплайна"""
            if len(points) > self.max_vertices:
                return points
            
            # Вычисление промежуточных точек
            p_start = evaluate_spline(t_start)
            p_end = evaluate_spline(t_end)
            p_mid = evaluate_spline((t_start + t_end) / 2)
            
            # Вычисление расстояния от средней точки до хорды
            chord = LineString([p_start, p_end])
            distance = chord.distance(Point(p_mid))
            
            # Если отклонение больше допустимого или слишком мало сегментов - разбиваем
            if distance > self.tolerance or (t_end - t_start) > 0.5:
                mid_t = (t_start + t_end) / 2
                recursive_subdivision(t_start, mid_t, points)
                if len(points) <= self.max_vertices:
                    recursive_subdivision(mid_t, t_end, points)
            else:
                if len(points) == 0 or points[-1] != p_start:
                    points.append(p_start)
                points.append(p_end)
            
            return points
        
        # Начальные параметры
        t_min = 0.0
        t_max = 1.0
        
        vertices = recursive_subdivision(t_min, t_max, [])
        
        # Замыкание контура для замкнутых сплайнов
        if is_closed and vertices and vertices[0] != vertices[-1]:
            vertices.append(vertices[0])
        
        return vertices
    
    def _create_shape_from_contour(self, contour: List[Tuple[float, float]], 
                                  name: str, layer: str) -> PolygonShape:
        """
        Создание объекта PolygonShape из контура
        
        Согласно разделу 2.1.5, обеспечивает корректное представление
        многосвязных областей с отверстиями.
        """
        # Удаление повторяющихся точек в конце контура
        if len(contour) > 1 and contour[0] == contour[-1]:
            contour = contour[:-1]
        
        # Создание базовой фигуры
        shape = PolygonShape(contour, name=name)
        
        # Добавление метаданных
        shape.layer = layer
        shape.source = "DXF"
        
        shape.original_position = shape.centroid.copy()
        
        # Проверка на вырожденные случаи
        if shape.polygon.area < self.tolerance ** 2:
            logger.warning(f"Пропуск вырожденной фигуры {name} с площадью {shape.polygon.area:.6f} мм²")
            raise ValueError("Вырожденная геометрия")
        
        return shape
    
    def _process_composite_geometries(self, shapes: List[PolygonShape]) -> List[PolygonShape]:
        """
        Обработка составных геометрий и отверстий
        
        Реализует методы из раздела 2.1.5 для определения:
        - Внешних и внутренних контуров
        - Многосвязных областей
        - Составных фигур (несколько компонент)
        
        Алгоритм:
        1. Группировка фигур по пространственной близости
        2. Определение вложенности контуров
        3. Формирование многосвязных полигонов
        """
        logger.info("Обработка составных геометрий и отверстий...")
        start_time = time.time()
        
        if not shapes:
            return shapes
        
        # Создание пространственного индекса для ускорения поиска
        from rtree import index
        idx = index.Index()
        for i, shape in enumerate(shapes):
            minx, miny, maxx, maxy = shape.polygon.bounds
            idx.insert(i, (minx, miny, maxx, maxy))
        
        processed_shapes = []
        used_indices = set()
        
        for i, shape in enumerate(shapes):
            if i in used_indices:
                continue
            
            # Поиск потенциально вложенных контуров
            minx, miny, maxx, maxy = shape.polygon.bounds
            candidates = list(idx.intersection((minx, miny, maxx, maxy)))
            
            # Определение внешнего контура (самый большой по площади)
            outer_contour = shape
            inner_contours = []
            
            for j in candidates:
                if j in used_indices or j == i:
                    continue
                
                candidate = shapes[j]
                
                # Проверка вложенности: кандидат внутри внешнего контура
                if outer_contour.polygon.contains(candidate.polygon):
                    # Проверка на отверстие (направление обхода)
                    if self._is_hole_contour(candidate):
                        inner_contours.append(candidate)
                        used_indices.add(j)
            
            # Создание составной фигуры
            if inner_contours:
                inner_contours_data = [contour.outer_contour for contour in inner_contours]
                composite_shape = PolygonShape(
                    outer_contour.outer_contour,
                    inner_contours_data,
                    name=f"{outer_contour.name}_composite"
                )
                composite_shape.layer = outer_contour.layer
                composite_shape.source = "DXF_composite"
                processed_shapes.append(composite_shape)
                used_indices.add(i)
            else:
                processed_shapes.append(shape)
                used_indices.add(i)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Обработано {len(processed_shapes)} составных фигур за {elapsed_time:.2f} секунд")
        return processed_shapes
    
    def _is_hole_contour(self, shape: PolygonShape) -> bool:
        """
        Определение, является ли контур отверстием (внутренним контуром)
        
        В DXF внутренние контуры (отверстия) обычно имеют обратное направление обхода.
        """
        # Расчет ориентации контура по площади (формула Гаусса)
        contour = shape.outer_contour
        area = 0.0
        n = len(contour)
        
        for i in range(n):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % n]
            area += (x2 - x1) * (y2 + y1)
        
        # При положительной площади - контур по часовой стрелке (обычно отверстия)
        return area > 0
    
    def _extract_block(self, insert_entity, doc: ezdxf.document.Drawing) -> List[PolygonShape]:
        """
        Извлечение фигур из блока (INSERT)
        
        Блоки в DXF могут содержать несколько геометрических объектов.
        Этот метод рекурсивно извлекает все фигуры из блока.
        """
        shapes = []
        block_name = insert_entity.dxf.name
        
        try:
            block = doc.blocks.get(block_name)
            if block is None:
                logger.warning(f"Блок '{block_name}' не найден в определении блоков")
                return shapes
            
            for entity in block:
                try:
                    entity_type = entity.dxftype()
                    
                    if entity_type == 'LWPOLYLINE':
                        contour = self._extract_lwpolyline(entity)
                        if contour and len(contour) >= 3:
                            # Применение трансформации блока (масштаб, поворот, смещение)
                            transformed_contour = self._transform_block_contour(
                                contour, insert_entity
                            )
                            shape = self._create_shape_from_contour(
                                transformed_contour, f"block_{block_name}_{len(shapes)}", 
                                insert_entity.dxf.layer
                            )
                            shapes.append(shape)
                    
                    elif entity_type == 'POLYLINE':
                        contour = self._extract_polyline(entity)
                        if contour and len(contour) >= 3:
                            transformed_contour = self._transform_block_contour(
                                contour, insert_entity
                            )
                            shape = self._create_shape_from_contour(
                                transformed_contour, f"block_{block_name}_{len(shapes)}",
                                insert_entity.dxf.layer
                            )
                            shapes.append(shape)
                    
                    elif entity_type == 'CIRCLE':
                        contour = self._extract_circle(entity)
                        transformed_contour = self._transform_block_contour(
                            contour, insert_entity
                        )
                        shape = self._create_shape_from_contour(
                            transformed_contour, f"block_{block_name}_{len(shapes)}",
                            insert_entity.dxf.layer
                        )
                        shapes.append(shape)
                    
                    elif entity_type == 'ARC':
                        contour = self._extract_arc(entity)
                        if contour and len(contour) >= 3:
                            transformed_contour = self._transform_block_contour(
                                contour, insert_entity
                            )
                            shape = self._create_shape_from_contour(
                                transformed_contour, f"block_{block_name}_{len(shapes)}",
                                insert_entity.dxf.layer
                            )
                            shapes.append(shape)
                    
                    elif entity_type == 'INSERT':
                        nested_shapes = self._extract_block(entity, doc)
                        shapes.extend(nested_shapes)
                
                except ValueError as e:
                    continue
                except Exception as e:
                    logger.warning(f"Ошибка при обработке объекта {entity_type} в блоке {block_name}: {e}")
                    continue
            
            line_entities = [e for e in block if e.dxftype() == 'LINE']
            if line_entities:
                line_shapes = self._extract_lines_from_block(line_entities, insert_entity, block_name)
                shapes.extend(line_shapes)
        
        except Exception as e:
            logger.warning(f"Ошибка при извлечении блока '{block_name}': {e}")
        
        return shapes
    
    def _extract_lines_from_block(self, line_entities, insert_entity, block_name: str) -> List[PolygonShape]:
        """
        Группировка LINE объектов в замкнутые контуры
        
        LINE объекты могут образовывать прямоугольники или другие замкнутые фигуры,
        если их конечные точки совпадают.
        """
        shapes = []
        
        if not line_entities:
            return shapes
        
        lines = []
        for line_entity in line_entities:
            start = (line_entity.dxf.start.x, line_entity.dxf.start.y)
            end = (line_entity.dxf.end.x, line_entity.dxf.end.y)
            lines.append((start, end))
        
        def find_closed_contour(start_line_idx, visited_lines):
            """Поиск замкнутого контура, начиная с заданной линии"""
            contour = []
            current_line_idx = start_line_idx
            start_point = lines[current_line_idx][0]
            current_point = lines[current_line_idx][1]
            contour.append(start_point)
            contour.append(current_point)
            visited_lines.add(current_line_idx)
            
            max_iterations = len(lines)
            iterations = 0
            
            while iterations < max_iterations:
                iterations += 1
                found_next = False
                
                for i, (line_start, line_end) in enumerate(lines):
                    if i in visited_lines:
                        continue
                    
                    tolerance = 0.01
                    if abs(line_start[0] - current_point[0]) < tolerance and abs(line_start[1] - current_point[1]) < tolerance:
                        # Линия начинается в текущей точке
                        current_point = line_end
                        contour.append(current_point)
                        visited_lines.add(i)
                        found_next = True
                        break
                    elif abs(line_end[0] - current_point[0]) < tolerance and abs(line_end[1] - current_point[1]) < tolerance:
                        # Линия заканчивается в текущей точке
                        current_point = line_start
                        contour.append(current_point)
                        visited_lines.add(i)
                        found_next = True
                        break
                
                if not found_next:
                    break
                
                if abs(current_point[0] - start_point[0]) < tolerance and abs(current_point[1] - start_point[1]) < tolerance:
                    if len(contour) >= 4:  # Минимум 4 точки для прямоугольника
                        if len(contour) > 1 and abs(contour[-1][0] - contour[0][0]) < tolerance and abs(contour[-1][1] - contour[0][1]) < tolerance:
                            contour = contour[:-1]
                        return contour
                    break
            
            return None
        
        visited_lines = set()
        contour_counter = 0
        
        for i in range(len(lines)):
            if i in visited_lines:
                continue
            
            contour = find_closed_contour(i, visited_lines)
            if contour and len(contour) >= 3:
                transformed_contour = self._transform_block_contour(contour, insert_entity)
                
                try:
                    shape = self._create_shape_from_contour(
                        transformed_contour, f"block_{block_name}_line_{contour_counter}",
                        insert_entity.dxf.layer
                    )
                    shapes.append(shape)
                    contour_counter += 1
                except ValueError:
                    continue
        
        return shapes
    
    def _transform_block_contour(self, contour: List[Tuple[float, float]], 
                                insert_entity) -> List[Tuple[float, float]]:
        """
        Применение трансформации блока к контуру (масштаб, поворот, смещение)
        """
        insert_point = (insert_entity.dxf.insert.x, insert_entity.dxf.insert.y)
        scale_x = insert_entity.dxf.xscale if hasattr(insert_entity.dxf, 'xscale') else 1.0
        scale_y = insert_entity.dxf.yscale if hasattr(insert_entity.dxf, 'yscale') else 1.0
        rotation = math.radians(insert_entity.dxf.rotation) if hasattr(insert_entity.dxf, 'rotation') else 0.0
        
        transformed_contour = []
        for x, y in contour:
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            
            x_rotated = x_scaled * math.cos(rotation) - y_scaled * math.sin(rotation)
            y_rotated = x_scaled * math.sin(rotation) + y_scaled * math.cos(rotation)
            
            x_final = x_rotated + insert_point[0]
            y_final = y_rotated + insert_point[1]
            
            transformed_contour.append((x_final, y_final))
        
        return transformed_contour
    
    def _extract_defects_from_block(self, insert_entity, doc: ezdxf.document.Drawing, 
                                   layer_name: str) -> List[PolygonShape]:
        """
        Извлечение дефектных зон из блока (INSERT) с правильным применением трансформаций
        
        :param insert_entity: INSERT entity из modelspace
        :param doc: DXF документ
        :param layer_name: Имя слоя дефектных зон
        :return: Список дефектных зон с примененными трансформациями
        """
        defect_shapes = []
        block_name = insert_entity.dxf.name
        
        try:
            rotation_deg = insert_entity.dxf.rotation if hasattr(insert_entity.dxf, 'rotation') else 0.0
            logger.debug(f"Извлечение дефектных зон из блока '{block_name}' (поворот: {rotation_deg:.2f}°)")
            
            block = doc.blocks.get(block_name)
            if block is None:
                logger.warning(f"Блок '{block_name}' не найден при извлечении дефектных зон")
                return defect_shapes
            
            for entity in block:
                try:
                    entity_layer = entity.dxf.layer
                    if entity_layer != layer_name and insert_entity.dxf.layer != layer_name:
                        continue
                    
                    entity_type = entity.dxftype()
                    
                    if entity_type == 'LWPOLYLINE':
                        contour = self._extract_lwpolyline(entity)
                        if contour and len(contour) >= 3:
                            transformed_contour = self._transform_block_contour(
                                contour, insert_entity
                            )
                            defect_shape = self._create_shape_from_contour(
                                transformed_contour, f"defect_block_{block_name}_{len(defect_shapes)}", 
                                layer_name
                            )
                            defect_shapes.append(defect_shape)
                            logger.debug(f"  Извлечена дефектная зона из LWPOLYLINE в блоке '{block_name}'")
                    
                    elif entity_type == 'CIRCLE':
                        contour = self._extract_circle(entity)
                        transformed_contour = self._transform_block_contour(
                            contour, insert_entity
                        )
                        defect_shape = self._create_shape_from_contour(
                            transformed_contour, f"defect_block_{block_name}_{len(defect_shapes)}",
                            layer_name
                        )
                        defect_shapes.append(defect_shape)
                        logger.debug(f"  Извлечена дефектная зона из CIRCLE в блоке '{block_name}'")
                    
                    elif entity_type == 'INSERT':
                        nested_defects = self._extract_defects_from_block(entity, doc, layer_name)
                        defect_shapes.extend(nested_defects)
                
                except ValueError as e:
                    continue
                except Exception as e:
                    logger.warning(f"Ошибка при обработке дефектной зоны в блоке {block_name}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Ошибка при извлечении дефектных зон из блока '{block_name}': {e}")
        
        return defect_shapes
    
    def _extract_technological_constraints(self, doc: ezdxf.document.Drawing, shapes: List[PolygonShape]):
        """
        Извлечение технологических ограничений из DXF файла
        
        Согласно разделу 2.5, включает:
        - Минимальные зазоры
        - Ограничения на ориентацию
        - Приоритеты резки
        - Дефектные зоны
        """
        logger.info("Извлечение технологических ограничений...")
        
        # 1. Извлечение ограничений из текстовых объектов на специальных слоях
        constraint_layers = [layer for layer, semantic_type in self.layer_mapping.items() 
                           if semantic_type == 'constraints']
        
        for layer_name in constraint_layers:
            for entity in doc.modelspace().query(f'TEXT[layer=="{layer_name}"]'):
                try:
                    text = entity.dxf.text.strip()
                    if ':' in text:
                        identifier, constraint_str = text.split(':', 1)
                        identifier = identifier.strip()
                        constraint_str = constraint_str.strip()
                        
                        # Поиск соответствующей фигуры
                        for shape in shapes:
                            matched = False
                            
                            if identifier.lower() in shape.name.lower():
                                matched = True
                            else:
                                import re
                                identifier_number_match = re.search(r'(\d+)', identifier)
                                shape_number_match = re.search(r'(\d+)', shape.name)
                                
                                if identifier_number_match and shape_number_match:
                                    # Сопоставление по номеру
                                    if identifier_number_match.group(1) == shape_number_match.group(1):
                                        matched = True
                            
                            if matched:
                                self._apply_constraint_to_shape(shape, constraint_str)
                                break
                except Exception as e:
                    logger.warning(f"Ошибка при обработке ограничения '{text}': {e}")
        
        # 2. Извлечение дефектных зон
        defect_shapes = []
        defect_layers = [layer for layer, semantic_type in self.layer_mapping.items() 
                        if semantic_type == 'defects']
        
        for layer_name in defect_layers:
            for entity in doc.modelspace().query(f'*[layer=="{layer_name}"]'):
                try:
                    if entity.dxftype() == 'LWPOLYLINE':
                        contour = self._extract_lwpolyline(entity)
                        if contour and len(contour) >= 3:
                            defect_shape = self._create_shape_from_contour(
                                contour, f"defect_{len(defect_shapes)}", layer_name
                            )
                            defect_shapes.append(defect_shape)
                    
                    elif entity.dxftype() == 'CIRCLE':
                        contour = self._extract_circle(entity)
                        defect_shape = self._create_shape_from_contour(
                            contour, f"defect_{len(defect_shapes)}", layer_name
                        )
                        defect_shapes.append(defect_shape)
                    
                    elif entity.dxftype() == 'INSERT':
                        block_defects = self._extract_defects_from_block(entity, doc, layer_name)
                        if block_defects:
                            logger.info(f"  Извлечено {len(block_defects)} дефектных зон из блока '{entity.dxf.name}'")
                        defect_shapes.extend(block_defects)
                except Exception as e:
                    logger.warning(f"Ошибка при обработке дефектной зоны: {e}")
        
        if defect_shapes:
            logger.info(f"Извлечено {len(defect_shapes)} дефектных зон")
        
        # 3. Извлечение границ листа
        sheet_boundary = None
        boundary_layers = [layer for layer, semantic_type in self.layer_mapping.items() 
                          if semantic_type == 'sheet_boundary']
        
        for layer_name in boundary_layers:
            for entity in doc.modelspace().query(f'*[layer=="{layer_name}"]'):
                if entity.dxftype() == 'LWPOLYLINE':
                    contour = self._extract_lwpolyline(entity)
                    if contour and len(contour) >= 3:
                        polygon = Polygon(contour)
                        if polygon.is_valid and polygon.area > 0:
                            sheet_boundary = polygon
                            break
        
        # 4. Сохранение извлеченных данных в атрибуты класса
        self.technological_constraints = {
            'defect_zones': defect_shapes,
            'sheet_boundary': sheet_boundary,
            'material_type': self.attributes.get('MATERIAL', 'steel'),
            'cutting_technology': self.attributes.get('CUTTING_TECH', 'laser')
        }
    
    def _apply_constraint_to_shape(self, shape: PolygonShape, constraint_str: str):
        """
        Применение технологического ограничения к фигуре
        
        Поддерживаемые форматы:
        - priority=1 (приоритет размещения)
        - angles=0,90 (допустимые углы поворота)
        - orientation=fixed (фиксированная ориентация)
        - anisotropic=true (анизотропный материал)
        """
        try:
            if constraint_str.startswith('priority='):
                priority = int(constraint_str.split('=')[1].strip())
                shape.priority = priority
                logger.debug(f"Установлен приоритет {priority} для фигуры {shape.name}")
            
            elif constraint_str.startswith('angles='):
                angles_str = constraint_str.split('=')[1].strip().split(',')
                allowed_angles = [float(angle.strip()) % 360 for angle in angles_str]
                shape.allowed_angles = sorted(set(allowed_angles))
                logger.debug(f"Установлены допустимые углы {allowed_angles} для фигуры {shape.name}")
            
            elif constraint_str.startswith('orientation=fixed'):
                shape.allowed_angles = [0.0]
                logger.debug(f"Установлена фиксированная ориентация для фигуры {shape.name}")
            
            elif constraint_str.startswith('anisotropic=true'):
                shape.is_anisotropic = True
                logger.debug(f"Фигура {shape.name} помечена как анизотропный материал")
        
        except Exception as e:
            logger.warning(f"Ошибка при применении ограничения '{constraint_str}' к фигуре {shape.name}: {e}")
    
    def get_sheet_boundary(self) -> Optional[Polygon]:
        """Получение границ листа из DXF файла"""
        return self.technological_constraints.get('sheet_boundary')
    
    def get_sheet_size(self) -> Optional[Tuple[float, float]]:
        """
        Получение размеров листа из границы (BOUNDARY слой)
        
        :return: Кортеж (ширина, высота) в мм или None если граница не найдена
        """
        sheet_boundary = self.get_sheet_boundary()
        if sheet_boundary is not None:
            bounds = sheet_boundary.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            logger.info(f"Размеры листа из BOUNDARY слоя: {width:.2f}x{height:.2f} мм")
            return (width, height)
        return None
    
    def get_defect_zones(self) -> List[PolygonShape]:
        """Получение списка дефектных зон"""
        return self.technological_constraints.get('defect_zones', [])
    
    def get_technological_metadata(self) -> Dict[str, Any]:
        """Получение технологических метаданных"""
        metadata = self.attributes.copy()
        metadata.update({
            'tolerance': self.tolerance,
            'cutting_technology': self.technological_constraints.get('cutting_technology', 'unknown'),
            'material_type': self.technological_constraints.get('material_type', 'unknown')
        })
        return metadata

def import_dxf(filename: str, tolerance: float = 0.1, max_vertices: int = 200, 
               return_importer: bool = False) -> List[PolygonShape]:
    """
    Функция-обертка для удобного импорта DXF файлов
    
    :param filename: Путь к DXF файлу
    :param tolerance: Точность аппроксимации кривых в мм
    :param max_vertices: Максимальное число вершин для одной фигуры
    :param return_importer: Если True, возвращает также экземпляр импортера для доступа к метаданным
    :return: Список импортированных фигур (или кортеж (shapes, importer) если return_importer=True)
    """
    importer = DXFImporter(tolerance=tolerance, max_vertices=max_vertices)
    shapes = importer.import_file(filename)
    
    if return_importer:
        return shapes, importer
    return shapes

def get_sheet_size_from_dxf(filename: str) -> Optional[Tuple[float, float]]:
    """
    Получение размера листа из DXF файла (из BOUNDARY слоя)
    
    :param filename: Путь к DXF файлу
    :return: Кортеж (ширина, высота) в мм или None если граница не найдена
    """
    importer = DXFImporter(tolerance=0.1, max_vertices=200)
    try:
        importer.import_file(filename)  # Загружаем файл для извлечения границы
        return importer.get_sheet_size()
    except Exception as e:
        logger.warning(f"Не удалось получить размер листа из DXF: {e}")
        return None