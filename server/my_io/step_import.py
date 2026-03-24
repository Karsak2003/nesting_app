import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Any
from core.geometry import PolygonShape
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import time
import logging
import os

# Попытка импорта pythonocc-core для работы с STEP
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
    from OCC.Core.TopoDS import TopoDS_Shape, topods_Face, topods_Wire, topods_Edge
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Geom import Geom_Curve, Geom_BSplineCurve, Geom_Circle, Geom_Ellipse, Geom_TrimmedCurve
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepTools import breptools_UVBounds
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
    from OCC.Core.Standard import Standard_NullValue
    from OCC.Core.Precision import precision_Angular, precision_Confusion
    
    STEP_SUPPORT = True
except ImportError as e:
    logging.warning(f"Не удалось импортировать pythonocc-core: {e}")
    logging.warning("Импорт из STEP будет недоступен. Установите: pip install pythonocc-core")
    STEP_SUPPORT = False

# Настройка логирования
logger = logging.getLogger(__name__)

class STEPImporter:
    """
    Класс для импорта геометрических данных из STEP файлов (ISO 10303-21)
    
    Реализует функционал, описанный в главах 2 и 3:
    - Поддержка NURBS-геометрии и адаптивная полигонализация
    - Извлечение многосвязных областей с отверстиями
    - Обработка составных геометрий
    - Извлечение метаданных и технологических ограничений
    - Соответствие промышленным стандартам (ГОСТ Р ИСО 10303-21)
    """
    
    if STEP_SUPPORT:
        def __init__(self, tolerance: float = 0.1, max_vertices: int = 200, angular_deflection: float = 0.1):
            """
            Инициализация STEP импортера
            
            :param tolerance: Точность аппроксимации кривых в мм (согласно разделу 2.1.7)
            :param max_vertices: Максимальное число вершин для одной фигуры
            :param angular_deflection: Угловое отклонение для полигонализации (в градусах)
            """
            self.tolerance = tolerance
            self.max_vertices = max_vertices
            self.angular_deflection = math.radians(angular_deflection)  # Преобразование в радианы
            self.metadata = {}
            self.sheet_size = None
            
        def import_file(self, filename: str, sheet_size: Optional[Tuple[float, float]] = None) -> List[PolygonShape]:
            """
            Импорт фигур из STEP файла
            
            :param filename: Путь к STEP файлу
            :param sheet_size: Размеры листа (ширина, высота) для нормализации координат
            :return: Список импортированных фигур
            """
            start_time = time.time()
            logger.info(f"Начало импорта STEP файла: {filename}")
            
            if not STEP_SUPPORT:
                raise ImportError("Модуль pythonocc-core не установлен. Импорт из STEP недоступен.")
            
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Файл не найден: {filename}")
            
            try:
                # Загрузка STEP документа
                step_reader = STEPControl_Reader()
                status = step_reader.ReadFile(filename)
                
                if status != IFSelect_RetDone:
                    raise ValueError("Ошибка чтения STEP файла")
                
                step_reader.TransferRoots()
                shape = step_reader.Shape()
                
                logger.info(f"STEP файл успешно загружен. Форма получена.")
                
                # Извлечение метаданных
                self._extract_metadata(step_reader)
                
                # Извлечение фигур
                shapes = self._extract_shapes(shape)
                
                # Обработка составных фигур и отверстий
                shapes = self._process_composite_geometries(shapes)
                
                # Нормализация координат относительно размеров листа
                if sheet_size:
                    shapes = self._normalize_coordinates(shapes, sheet_size)
                    self.sheet_size = sheet_size
                
                elapsed_time = time.time() - start_time
                logger.info(f"Импорт STEP завершен успешно. Загружено {len(shapes)} фигур за {elapsed_time:.2f} секунд")
                
                return shapes
                
            except Exception as e:
                logger.error(f"Ошибка при импорте STEP файла {filename}: {e}")
                raise
        
        def _extract_metadata(self, step_reader):
            """
            Извлечение метаданных из STEP файла
            
            Согласно разделу 2.5.6, важна интеграция с промышленными стандартами,
            включая извлечение технологических параметров из метаданных.
            """
            logger.info("Извлечение метаданных из STEP файла...")
            
            try:
                # Извлечение имени файла
                self.metadata['filename'] = step_reader.FileName().split('/')[-1]
                
                # Извлечение информации о единицах измерения
                self.metadata['units'] = "mm"  # По умолчанию в STEP используются мм
                
                # Извлечение информации о материале, если доступно
                # В STEP эта информация может быть в атрибутах
                self.metadata['material'] = "steel"  # Значение по умолчанию
                
                # Извлечение технологических параметров
                self.metadata['cutting_technology'] = "laser"  # По умолчанию
                
                logger.info(f"Метаданные извлечены: {self.metadata}")
            except Exception as e:
                logger.warning(f"Не удалось извлечь метаданные: {e}")
        
        def _extract_shapes(self, shape: TopoDS_Shape) -> List[PolygonShape]:
            """
            Извлечение фигур из геометрической формы STEP
            
            Обрабатывает различные типы геометрии в соответствии с разделом 2.1.2 и 2.1.5:
            - Грани (faces) с внешними и внутренними контурами
            - Проволочные каркасы (wires)
            - Составные геометрии
            """
            logger.info("Извлечение фигур из геометрической формы STEP...")
            shapes = []
            shape_counter = 1
            
            # Обход всех граней в форме
            face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
            
            while face_explorer.More():
                face = topods_Face(face_explorer.Current())
                try:
                    # Извлечение внешнего контура грани
                    outer_wire = BRep_Tool.OuterWire(face)
                    if not outer_wire.IsNull():
                        outer_contour = self._extract_wire_contour(outer_wire)
                        
                        if outer_contour and len(outer_contour) >= 3:
                            # Извлечение внутренних контуров (отверстий)
                            inner_contours = []
                            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
                            
                            while wire_explorer.More():
                                wire = topods_Wire(wire_explorer.Current())
                                if not wire.IsSame(outer_wire):
                                    inner_contour = self._extract_wire_contour(wire)
                                    if inner_contour and len(inner_contour) >= 3:
                                        inner_contours.append(inner_contour)
                                wire_explorer.Next()
                            
                            # Создание фигуры
                            shape_obj = PolygonShape(
                                outer_contour,
                                inner_contours if inner_contours else None,
                                name=f"step_shape_{shape_counter}"
                            )
                            shape_obj.source = "STEP"
                            
                            # Проверка на вырожденные случаи
                            if shape_obj.polygon.area >= self.tolerance ** 2:
                                shapes.append(shape_obj)
                                shape_counter += 1
                            else:
                                logger.warning(f"Пропущена вырожденная фигура {shape_obj.name} с площадью {shape_obj.polygon.area:.6f} мм²")
                
                except Exception as e:
                    logger.warning(f"Ошибка при обработке грани: {e}")
                
                face_explorer.Next()
            
            logger.info(f"Извлечено {len(shapes)} фигур из STEP файла")
            return shapes
        
        def _extract_wire_contour(self, wire: TopoDS_Shape) -> List[Tuple[float, float]]:
            """
            Извлечение контура из проволочного каркаса (wire)
            
            Обрабатывает различные типы кривых, включая NURBS, окружности и эллипсы,
            с адаптивной полигонализацией для достижения заданной точности.
            """
            contour = []
            edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
            
            # Сбор всех сегментов контура
            edge_segments = []
            
            while edge_explorer.More():
                edge = topods_Edge(edge_explorer.Current())
                try:
                    # Извлечение кривой из ребра
                    curve_handle, first_param, last_param = BRep_Tool.Curve(edge)
                    
                    if curve_handle:
                        # Адаптивная полигонализация кривой
                        curve_points = self._adaptive_curve_polyline(curve_handle, first_param, last_param)
                        if curve_points:
                            edge_segments.append(curve_points)
                except Exception as e:
                    logger.warning(f"Ошибка при обработке ребра: {e}")
                
                edge_explorer.Next()
            
            # Соединение сегментов в единый контур
            if edge_segments:
                # Сортировка сегментов для правильного соединения
                sorted_segments = self._sort_edge_segments(edge_segments)
                
                # Формирование контура
                for i, segment in enumerate(sorted_segments):
                    if i == 0:
                        contour.extend(segment)
                    else:
                        # Пропускаем первую точку, так как она совпадает с последней точкой предыдущего сегмента
                        contour.extend(segment[1:])
                
                # Замыкание контура, если необходимо
                if len(contour) > 1 and contour[0] != contour[-1]:
                    contour.append(contour[0])
            
            return contour
        
        def _adaptive_curve_polyline(self, curve_handle, first_param: float, last_param: float) -> List[Tuple[float, float]]:
            """
            Адаптивная полигонализация кривой с заданной точностью
            
            Реализует алгоритм из раздела 2.1.7 для обработки NURBS-кривых
            с точностью ≤ 0.1 мм, что соответствует требованиям промышленных стандартов.
            """
            try:
                # Создание алгоритма для равномерной дискретизации с заданным отклонением
                discretizer = GCPnts_QuasiUniformDeflection()
                discretizer.Initialize(curve_handle, self.tolerance, first_param, last_param)
                
                if discretizer.IsDone() and discretizer.NbPoints() > 0:
                    points = []
                    for i in range(1, discretizer.NbPoints() + 1):
                        p = discretizer.Value(i)
                        points.append((p.X(), p.Y()))
                    return points
                else:
                    logger.warning("Не удалось выполнить дискретизацию кривой")
                    return []
            
            except Exception as e:
                logger.error(f"Ошибка при полигонализации кривой: {e}")
                return []
        
        def _sort_edge_segments(self, segments: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
            """
            Сортировка сегментов контура для правильного соединения
            
            Обеспечивает правильное соединение сегментов в замкнутый контур
            даже при произвольном порядке обхода в STEP файле.
            """
            if not segments:
                return []
            
            # Начинаем с первого сегмента
            sorted_segments = [segments[0]]
            remaining_segments = segments[1:]
            
            while remaining_segments:
                last_segment = sorted_segments[-1]
                last_point = last_segment[-1]
                
                found = False
                for i, segment in enumerate(remaining_segments):
                    # Проверка на соединение с началом сегмента
                    if math.hypot(last_point[0] - segment[0][0], last_point[1] - segment[0][1]) < self.tolerance:
                        sorted_segments.append(segment)
                        remaining_segments.pop(i)
                        found = True
                        break
                    
                    # Проверка на соединение с концом сегмента (требуется разворот)
                    if math.hypot(last_point[0] - segment[-1][0], last_point[1] - segment[-1][1]) < self.tolerance:
                        reversed_segment = segment[::-1]
                        sorted_segments.append(reversed_segment)
                        remaining_segments.pop(i)
                        found = True
                        break
                
                # Если не найдено соединение, прерываем цикл
                if not found:
                    logger.warning("Не удалось соединить все сегменты в единый контур")
                    break
            
            return sorted_segments
        
        def _process_composite_geometries(self, shapes: List[PolygonShape]) -> List[PolygonShape]:
            """
            Обработка составных геометрий и отверстий
            
            Согласно разделу 2.1.5, обеспечивает корректное представление
            многосвязных областей с отверстиями для промышленных деталей.
            """
            logger.info("Обработка составных геометрий и отверстий...")
            
            if not shapes:
                return shapes
            
            # Группировка фигур по пространственной близости
            processed_shapes = []
            used_indices = set()
            
            # Создание пространственного индекса для ускорения поиска
            from rtree import index
            idx = index.Index()
            for i, shape in enumerate(shapes):
                minx, miny, maxx, maxy = shape.polygon.bounds
                idx.insert(i, (minx, miny, maxx, maxy))
            
            for i, shape in enumerate(shapes):
                if i in used_indices:
                    continue
                
                # Поиск потенциально связанных фигур
                minx, miny, maxx, maxy = shape.polygon.bounds
                search_box = (
                    minx - self.tolerance,
                    miny - self.tolerance,
                    maxx + self.tolerance,
                    maxy + self.tolerance
                )
                candidates = list(idx.intersection(search_box))
                
                # Определение главной фигуры и отверстий
                main_shape = shape
                holes = []
                
                for j in candidates:
                    if j in used_indices or j == i:
                        continue
                    
                    candidate = shapes[j]
                    
                    # Проверка, является ли кандидат отверстием в главной фигуре
                    if main_shape.polygon.contains(candidate.polygon):
                        holes.append(candidate)
                        used_indices.add(j)
                
                # Создание составной фигуры
                if holes:
                    inner_contours = [hole.outer_contour for hole in holes]
                    composite_shape = PolygonShape(
                        main_shape.outer_contour,
                        inner_contours,
                        name=f"{main_shape.name}_composite"
                    )
                    composite_shape.source = "STEP_composite"
                    processed_shapes.append(composite_shape)
                    used_indices.add(i)
                else:
                    processed_shapes.append(shape)
                    used_indices.add(i)
            
            logger.info(f"Обработано {len(processed_shapes)} составных фигур")
            return processed_shapes
        
        def _normalize_coordinates(self, shapes: List[PolygonShape], sheet_size: Tuple[float, float]) -> List[PolygonShape]:
            """
            Нормализация координат фигур относительно размеров листа
            
            Согласно разделу 2.4, требуется корректное позиционирование фигур
            в пределах заданной области листа.
            """
            logger.info("Нормализация координат относительно размеров листа...")
            
            if not shapes:
                return shapes
            
            # Расчет текущих границ всех фигур
            all_x = []
            all_y = []
            
            for shape in shapes:
                all_x.extend([point[0] for point in shape.outer_contour])
                all_y.extend([point[1] for point in shape.outer_contour])
                
                for inner_contour in shape.inner_contours:
                    all_x.extend([point[0] for point in inner_contour])
                    all_y.extend([point[1] for point in inner_contour])
            
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            current_width = max_x - min_x
            current_height = max_y - min_y
            
            # Коэффициенты масштабирования (сохраняем пропорции)
            scale_x = sheet_size[0] / current_width if current_width > 0 else 1.0
            scale_y = sheet_size[1] / current_height if current_height > 0 else 1.0
            scale = min(scale_x, scale_y)
            
            # Сдвиг для центрирования
            offset_x = (sheet_size[0] - current_width * scale) / 2.0
            offset_y = (sheet_size[1] - current_height * scale) / 2.0
            
            # Применение трансформации к каждой фигуре
            normalized_shapes = []
            for shape in shapes:
                # Трансформация внешнего контура
                normalized_outer = [
                    ((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y)
                    for x, y in shape.outer_contour
                ]
                
                # Трансформация внутренних контуров
                normalized_inners = []
                for inner_contour in shape.inner_contours:
                    normalized_inner = [
                        ((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y)
                        for x, y in inner_contour
                    ]
                    normalized_inners.append(normalized_inner)
                
                # Создание новой фигуры с нормализованными координатами
                normalized_shape = PolygonShape(
                    normalized_outer,
                    normalized_inners if normalized_inners else None,
                    name=shape.name
                )
                normalized_shape.source = f"{shape.source}_normalized"
                normalized_shapes.append(normalized_shape)
            
            logger.info(f"Координаты нормализованы. Масштаб: {scale:.4f}, сдвиг: ({offset_x:.2f}, {offset_y:.2f})")
            return normalized_shapes
        
        def get_technological_metadata(self) -> Dict[str, Any]:
            """
            Получение технологических метаданных из STEP файла
            
            Согласно разделу 2.5, включает информацию о:
            - Материале заготовки
            - Технологии резки
            - Ориентационных ограничениях
            - Приоритетах размещения
            """
            return {
                'filename': self.metadata.get('filename', ''),
                'material': self.metadata.get('material', 'unknown'),
                'cutting_technology': self.metadata.get('cutting_technology', 'unknown'),
                'units': self.metadata.get('units', 'mm'),
                'sheet_size': self.sheet_size,
                'tolerance': self.tolerance
            }
    
    else:
        # Заглушка для случая, когда pythonocc-core не установлен
        def __init__(self, tolerance: float = 0.1, max_vertices: int = 200):
            raise ImportError("Модуль pythonocc-core не установлен. Импорт из STEP недоступен. Установите: pip install pythonocc-core")
        
        def import_file(self, filename: str, sheet_size: Optional[Tuple[float, float]] = None) -> List[PolygonShape]:
            raise ImportError("Модуль pythonocc-core не установлен. Импорт из STEP недоступен. Установите: pip install pythonocc-core")

def import_step(filename: str, tolerance: float = 0.1, max_vertices: int = 200, 
               sheet_size: Optional[Tuple[float, float]] = None) -> List[PolygonShape]:
    """
    Функция-обертка для удобного импорта STEP файлов
    
    :param filename: Путь к STEP файлу
    :param tolerance: Точность аппроксимации кривых в мм
    :param max_vertices: Максимальное число вершин для одной фигуры
    :param sheet_size: Размеры листа (ширина, высота) для нормализации координат
    :return: Список импортированных фигур
    """
    importer = STEPImporter(tolerance=tolerance, max_vertices=max_vertices)
    return importer.import_file(filename, sheet_size)