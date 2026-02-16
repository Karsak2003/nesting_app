import ezdxf
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from core.geometry import PolygonShape
from core.constraints import ConstraintManager
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate, translate
import json
import os
import time
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection, PatchCollection

# Попытка импорта для STEP экспорта
try:
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static_SetCVal
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Wire
    from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
    from OCC.Core.Geom import Geom_Plane
    STEP_SUPPORT = True
except ImportError as e:
    logging.warning(f"Не удалось импортировать OCC для STEP экспорта: {e}")
    STEP_SUPPORT = False

# Настройка логирования
logger = logging.getLogger(__name__)

class ResultExporter:
    """
    Класс для экспорта результатов раскроя в различные форматы
    
    Реализует требования глав 2 и 3 диссертации:
    - Соответствие ГОСТ Р ИСО 10303-21 (STEP)
    - Соответствие ГОСТ 34029-2016 (технологические требования)
    - Поддержка промышленных CAM-систем (SprutCAM, SigmaNEST)
    - Сохранение метаданных и технологических ограничений
    - Визуализация результатов для оператора
    """
    
    def __init__(self, sheet_size: Tuple[float, float], min_gap: float = 1.0):
        """
        Инициализация экспортера
        
        :param sheet_size: Размеры листа (ширина, высота) в мм
        :param min_gap: Минимальный технологический зазор в мм
        """
        self.sheet_size = sheet_size
        self.min_gap = min_gap
        self.metadata = {
            'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'software': 'IAGI Nesting System v1.0',
            'standards': ['ГОСТ 34029-2016', 'ГОСТ Р ИСО 10303-21'],
            'material': 'steel',
            'cutting_technology': 'laser'
        }
    
    def export_to_dxf(self, placements: List[Dict[str, Any]], 
                     output_path: str, 
                     defect_zones: Optional[List[PolygonShape]] = None,
                     constraint_manager: Optional[ConstraintManager] = None,
                     include_cutting_sequence: bool = True):
        """
        Экспорт результатов в DXF формат для CAM-систем
        
        Реализует требования раздела 2.5.6 по совместимости с промышленными стандартами
        и раздела 3.6.4 по готовности к внедрению в CAD/CAM-системы.
        
        :param placements: Список размещенных фигур с их позициями и углами
        :param output_path: Путь для сохранения DXF файла
        :param defect_zones: Список дефектных зон на листе
        :param constraint_manager: Менеджер технологических ограничений
        :param include_cutting_sequence: Включить последовательность резки в экспорт
        """
        logger.info(f"Экспорт результатов в DXF: {output_path}")
        start_time = time.time()
        
        try:
            # Создание нового DXF документа
            doc = ezdxf.new('R2010')  # Используем версию 2010 для совместимости
            msp = doc.modelspace()
            
            # Настройка единиц измерения (миллиметры)
            doc.header['$INSUNITS'] = 4  # миллиметры
            
            # Экспорт границ листа
            self._export_sheet_boundary(msp)
            
            # Экспорт размещенных фигур
            self._export_placed_shapes(msp, placements, include_cutting_sequence)
            
            # Экспорт дефектных зон
            if defect_zones:
                self._export_defect_zones(msp, defect_zones)
            
            # Экспорт технологических ограничений как метаданных
            if constraint_manager:
                self._export_constraints_metadata(msp, constraint_manager)
            
            # Добавление текстовой информации
            self._add_export_metadata(msp)
            
            # Сохранение файла
            doc.saveas(output_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"DXF экспорт завершен успешно. Файл сохранен: {output_path}")
            logger.info(f"Экспортировано {len(placements)} фигур за {elapsed_time:.2f} секунд")
            
            # Генерация отчета
            self._generate_export_report(placements, output_path, 'DXF')
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте в DXF: {e}")
            raise
    
    def _export_sheet_boundary(self, msp):
        """Экспорт границ листа в DXF"""
        width, height = self.sheet_size
        
        # Создание прямоугольника границ листа
        boundary_points = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
            (0, 0)  # Замыкание контура
        ]
        
        # Добавление полилинии
        msp.add_lwpolyline(boundary_points, dxfattribs={
            'layer': 'SHEET_BOUNDARY',
            'color': 7,  # Белый/черный (зависит от фона)
            'linetype': 'CONTINUOUS',
            'lineweight': 0.5
        })
        
        # Добавление текстовой метки
        msp.add_text(f"Лист {width}x{height} мм", dxfattribs={
            'layer': 'SHEET_INFO',
            'height': 5.0,
            'insert': (width/2, height-10),
            'halign': ezdxf.const.CENTER,
            'valign': ezdxf.const.MIDDLE
        })
    
    def _export_placed_shapes(self, msp, placements, include_cutting_sequence):
        """Экспорт размещенных фигур в DXF"""
        # Группировка фигур по приоритетам (если доступно)
        if include_cutting_sequence and all('priority' in p for p in placements):
            placements = sorted(placements, key=lambda x: x.get('priority', 1))
        
        for i, placement in enumerate(placements):
            shape = placement['shape']
            position = placement['position']
            angle = placement['angle']
            name = placement.get('name', f'part_{i+1}')
            priority = placement.get('priority', 1)
            
            # Получение трансформированного контура
            transformed_shape = shape.apply_transformation(position, angle)
            
            # Экспорт внешнего контура
            self._export_contour(
                msp, 
                transformed_shape.outer_contour, 
                layer=f'PARTS_PRIORITY_{priority}',
                color=self._get_color_by_priority(priority),
                name=name,
                shape_id=i+1
            )
            
            # Экспорт внутренних контуров (отверстий)
            for j, inner_contour in enumerate(transformed_shape.inner_contours):
                self._export_contour(
                    msp,
                    inner_contour,
                    layer=f'HOLES_PRIORITY_{priority}',
                    color=self._get_color_by_priority(priority),
                    name=f"{name}_hole_{j+1}",
                    shape_id=i+1
                )
            
            # Добавление текстовой метки с информацией
            centroid = transformed_shape.centroid
            text_info = f"{name}\nP={priority}"
            msp.add_text(text_info, dxfattribs={
                'layer': 'PART_LABELS',
                'height': 2.0,
                'insert': (centroid[0], centroid[1]),
                'halign': ezdxf.const.CENTER,
                'valign': ezdxf.const.MIDDLE,
                'color': self._get_color_by_priority(priority)
            })
    
    def _export_contour(self, msp, contour, layer, color, name, shape_id):
        """Экспорт одного контура в DXF"""
        # Убедимся, что контур замкнут
        if len(contour) > 1 and not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        
        # Создание полилинии
        polyline = msp.add_lwpolyline(
            [(float(x), float(y)) for x, y in contour],
            dxfattribs={
                'layer': layer,
                'color': color,
                'linetype': 'CONTINUOUS'
            }
        )
        
        # Добавление XDATA для хранения метаданных
        try:
            if hasattr(polyline, 'set_xdata'):
                polyline.set_xdata('IAGI_METADATA', [
                    (1000, f'SHAPE_NAME={name}'),
                    (1070, shape_id)
                ])
        except (AttributeError, ValueError, TypeError):
            pass
    
    def _export_defect_zones(self, msp, defect_zones):
        """Экспорт дефектных зон в DXF"""
        for i, defect in enumerate(defect_zones):
            # Экспорт контура дефектной зоны
            self._export_contour(
                msp,
                defect.outer_contour,
                layer='DEFECT_ZONES',
                color=1,  # Красный
                name=f'defect_{i+1}',
                shape_id=i+1
            )
            
            # Добавление маркера в центр дефекта
            centroid = defect.centroid
            msp.add_circle(
                center=(centroid[0], centroid[1]),
                radius=2.0,
                dxfattribs={
                    'layer': 'DEFECT_MARKERS',
                    'color': 1  # Красный
                }
            )
    
    def _export_constraints_metadata(self, msp, constraint_manager):
        """Экспорт технологических ограничений как метаданных в DXF"""
        # Экспорт ограничений на ориентацию
        for shape_id, angles in constraint_manager.orientation_constraints.items():
            text = f"{shape_id}:angles={','.join(str(a) for a in angles)}"
            msp.add_text(text, dxfattribs={
                'layer': 'CONSTRAINTS',
                'height': 2.5,
                'insert': (10, self.sheet_size[1] - 30 - 15 * len(constraint_manager.orientation_constraints))
            })
        
        # Экспорт приоритетов
        for shape_id, priority in constraint_manager.placement_priorities.items():
            text = f"{shape_id}:priority={priority}"
            msp.add_text(text, dxfattribs={
                'layer': 'CONSTRAINTS',
                'height': 2.5,
                'insert': (10, self.sheet_size[1] - 60 - 15 * len(constraint_manager.placement_priorities))
            })
        
        # Экспорт типа резки
        msp.add_text(f"TECHNOLOGY:{constraint_manager.current_technology}", dxfattribs={
            'layer': 'CONSTRAINTS',
            'height': 3.0,
            'insert': (10, self.sheet_size[1] - 15)
        })
    
    def _add_export_metadata(self, msp):
        """Добавление метаданных экспорта в DXF"""
        width, height = self.sheet_size
        x_offset = width + 20
        y_offset = height - 20
        
        # Заголовок
        msp.add_text("ТЕХНОЛОГИЧЕСКАЯ КАРТА РАСКРОЯ", dxfattribs={
            'layer': 'METADATA',
            'height': 5.0,
            'insert': (x_offset, y_offset),
            'color': 2
        })
        
        # Основная информация
        metadata_items = [
            f"Дата экспорта: {self.metadata['export_date']}",
            f"Программное обеспечение: {self.metadata['software']}",
            f"Материал: {self.metadata['material']}",
            f"Технология резки: {self.metadata['cutting_technology']}",
            f"Минимальный зазор: {self.min_gap} мм",
            f"Стандарты: {', '.join(self.metadata['standards'])}"
        ]
        
        for i, item in enumerate(metadata_items):
            msp.add_text(item, dxfattribs={
                'layer': 'METADATA',
                'height': 3.0,
                'insert': (x_offset, y_offset - 20 - (i * 5)),
                'color': 7
            })
    
    def _get_color_by_priority(self, priority):
        """Получение цвета DXF по приоритету фигуры"""
        color_map = {
            1: 2,    # Зеленый для высокого приоритета
            2: 3,    # Синий для среднего приоритета
            3: 4,    # Бирюзовый для низкого приоритета
        }
        return color_map.get(priority, 5)  # Желтый по умолчанию
    
    def export_to_step(self, placements: List[Dict[str, Any]], 
                      output_path: str,
                      include_sheet: bool = True):
        """
        Экспорт результатов в STEP формат (ISO 10303-21)
        
        Реализует требования раздела 2.5.6 по совместимости со стандартом ISO 10303-21
        и поддержке обмена данными между CAD-системами.
        
        :param placements: Список размещенных фигур
        :param output_path: Путь для сохранения STEP файла
        :param include_sheet: Включить геометрию листа в экспорт
        """
        if not STEP_SUPPORT:
            logger.error("Экспорт в STEP невозможен: библиотека pythonocc-core не установлена")
            raise ImportError("Для экспорта в STEP требуется установить pythonocc-core")
        
        logger.info(f"Экспорт результатов в STEP: {output_path}")
        start_time = time.time()
        
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TopoDS import TopoDS_Compound
            from OCC.Core.BRep import BRep_Builder
            
            # Создание составного тела для всех фигур
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            
            # Экспорт фигур
            for i, placement in enumerate(placements):
                shape = placement['shape']
                position = placement['position']
                angle = placement['angle']
                
                # Получение трансформированного контура
                transformed_shape = shape.apply_transformation(position, angle)
                
                # Создание грани из внешнего контура
                face = self._create_face_from_contour(transformed_shape.outer_contour)
                if face:
                    builder.Add(compound, face)
                
                # Создание граней для внутренних контуров (отверстий)
                for inner_contour in transformed_shape.inner_contours:
                    hole_face = self._create_face_from_contour(inner_contour)
                    if hole_face:
                        builder.Add(compound, hole_face)
            
            # Экспорт границ листа
            if include_sheet:
                sheet_face = self._create_sheet_face()
                if sheet_face:
                    builder.Add(compound, sheet_face)
            
            # Настройка writer'а
            writer = STEPControl_Writer()
            Interface_Static_SetCVal("write.step.schema", "AP203")  # Используем AP203 схему
            
            # Добавление данных
            writer.Transfer(compound, STEPControl_AsIs)
            
            # Сохранение файла
            status = writer.Write(output_path)
            
            if status != IFSelect_RetDone:
                raise Exception(f"Ошибка записи STEP файла. Статус: {status}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"STEP экспорт завершен успешно. Файл сохранен: {output_path}")
            logger.info(f"Экспортировано {len(placements)} фигур за {elapsed_time:.2f} секунд")
            
            # Генерация отчета
            self._generate_export_report(placements, output_path, 'STEP')
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте в STEP: {e}")
            raise
    
    def _create_face_from_contour(self, contour):
        """Создание грани из контура для STEP экспорта"""
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.TopoDS import TopoDS_Wire
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
            
            # Создание полигона из точек контура
            polygon = BRepBuilderAPI_MakePolygon()
            
            # Убедимся, что контур замкнут
            contour_points = contour.copy()
            if not np.array_equal(contour_points[0], contour_points[-1]):
                contour_points = np.vstack([contour_points, contour_points[0]])
            
            # Добавление точек в полигон
            for point in contour_points:
                polygon.Add(gp_Pnt(float(point[0]), float(point[1]), 0.0))
            
            # Замыкание полигона
            polygon.Close()
            
            # Создание проволочного каркаса
            wire = polygon.Wire()
            
            if wire.IsNull():
                return None
            
            # Создание грани из проволочного каркаса
            face_builder = BRepBuilderAPI_MakeFace(wire)
            
            if face_builder.IsDone():
                return face_builder.Face()
            else:
                logger.warning("Не удалось создать грань из контура")
                return None
                
        except Exception as e:
            logger.warning(f"Ошибка при создании грани из контура: {e}")
            return None
    
    def _create_sheet_face(self):
        """Создание грани для листа материала"""
        try:
            width, height = self.sheet_size
            
            # Определение углов листа
            corners = [
                (0, 0, 0),
                (width, 0, 0),
                (width, height, 0),
                (0, height, 0)
            ]
            
            return self._create_face_from_contour(np.array(corners))
            
        except Exception as e:
            logger.warning(f"Ошибка при создании грани листа: {e}")
            return None
    
    def export_to_svg(self, placements: List[Dict[str, Any]], 
                     output_path: str,
                     defect_zones: Optional[List[PolygonShape]] = None,
                     show_labels: bool = True,
                     show_dimensions: bool = True):
        """
        Экспорт результатов в SVG формат для визуализации
        
        :param placements: Список размещенных фигур
        :param output_path: Путь для сохранения SVG файла
        :param defect_zones: Список дефектных зон на листе
        :param show_labels: Отображать ли текстовые метки
        :param show_dimensions: Отображать ли размеры
        """
        logger.info(f"Экспорт результатов в SVG: {output_path}")
        start_time = time.time()
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
            import xml.etree.ElementTree as ET
            
            # Создание фигуры matplotlib
            width_inch = self.sheet_size[0] / 25.4  # Конвертация мм в дюймы
            height_inch = self.sheet_size[1] / 25.4
            dpi = 96
            
            fig, ax = plt.subplots(figsize=(width_inch, height_inch), dpi=dpi)
            ax.set_xlim(0, self.sheet_size[0])
            ax.set_ylim(0, self.sheet_size[1])
            ax.set_aspect('equal')
            ax.set_title('Результаты раскроя')
            ax.set_xlabel('X (мм)')
            ax.set_ylabel('Y (мм)')
            
            # Экспорт границ листа
            sheet_rect = plt.Rectangle((0, 0), self.sheet_size[0], self.sheet_size[1],
                                     fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(sheet_rect)
            
            # Экспорт размещенных фигур
            patches = []
            colors = []
            
            # Сортировка по приоритету для визуального отображения
            sorted_placements = sorted(placements, key=lambda x: x.get('priority', 1))
            
            for placement in sorted_placements:
                shape = placement['shape']
                position = placement['position']
                angle = placement['angle']
                name = placement.get('name', 'part')
                priority = placement.get('priority', 1)
                
                # Получение трансформированного контура
                transformed_shape = shape.apply_transformation(position, angle)
                
                # Основной контур
                polygon = MplPolygon(transformed_shape.outer_contour, closed=True,
                                   fill=True, alpha=0.7,
                                   edgecolor=self._get_mpl_color_by_priority(priority))
                
                patches.append(polygon)
                colors.append(self._get_mpl_color_by_priority(priority, alpha=0.3))
                
                # Внутренние контуры (отверстия)
                for inner_contour in transformed_shape.inner_contours:
                    hole = MplPolygon(inner_contour, closed=True,
                                    fill=True, color='white',
                                    edgecolor=self._get_mpl_color_by_priority(priority),
                                    linewidth=1)
                    patches.append(hole)
                    colors.append('white')
                
                # Текстовые метки
                if show_labels:
                    centroid = transformed_shape.centroid
                    ax.text(centroid[0], centroid[1], name,
                           ha='center', va='center',
                           fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Создание коллекции полигонов
            p = PatchCollection(patches, match_original=True)
            ax.add_collection(p)
            
            # Экспорт дефектных зон
            if defect_zones:
                for defect in defect_zones:
                    defect_polygon = MplPolygon(defect.outer_contour, closed=True,
                                              fill=True, color='red', alpha=0.3,
                                              edgecolor='darkred', linewidth=1.5,
                                              linestyle='--')
                    ax.add_patch(defect_polygon)
                    
                    # Текстовая метка для дефекта
                    centroid = defect.centroid
                    ax.text(centroid[0], centroid[1], "ДЕФЕКТ",
                           ha='center', va='center',
                           fontsize=6, color='darkred',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Размеры листа
            if show_dimensions:
                # Горизонтальный размер
                ax.annotate('', xy=(0, -10), xytext=(self.sheet_size[0], -10),
                           arrowprops=dict(arrowstyle='<->', lw=1))
                ax.text(self.sheet_size[0]/2, -15, f"{self.sheet_size[0]:.0f} мм",
                       ha='center', va='top', fontsize=8)
                
                # Вертикальный размер
                ax.annotate('', xy=(-10, 0), xytext=(-10, self.sheet_size[1]),
                           arrowprops=dict(arrowstyle='<->', lw=1))
                ax.text(-15, self.sheet_size[1]/2, f"{self.sheet_size[1]:.0f} мм",
                       ha='right', va='center', rotation=90, fontsize=8)
            
            # Расчет коэффициента использования материала
            total_area = sum(placement['shape'].area for placement in placements)
            sheet_area = self.sheet_size[0] * self.sheet_size[1]
            utilization = total_area / sheet_area * 100
            
            # Информационная панель
            info_text = (f"Коэффициент использования: {utilization:.1f}%\n"
                        f"Число деталей: {len(placements)}\n"
                        f"Минимальный зазор: {self.min_gap} мм")
            
            ax.text(self.sheet_size[0] - 5, 5, info_text,
                   ha='right', va='bottom', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Решетка
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Сохранение в SVG
            fig.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close(fig)
            
            elapsed_time = time.time() - start_time
            logger.info(f"SVG экспорт завершен успешно. Файл сохранен: {output_path}")
            logger.info(f"Экспортировано {len(placements)} фигур за {elapsed_time:.2f} секунд")
            
            # Генерация отчета
            self._generate_export_report(placements, output_path, 'SVG')
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте в SVG: {e}")
            raise
    
    def _get_mpl_color_by_priority(self, priority, alpha=0.7):
        """Получение цвета matplotlib по приоритету фигуры"""
        color_map = {
            1: (0.0, 0.8, 0.0, alpha),    # Зеленый для высокого приоритета
            2: (0.0, 0.0, 1.0, alpha),    # Синий для среднего приоритета
            3: (0.0, 0.8, 0.8, alpha),    # Бирюзовый для низкого приоритета
        }
        return color_map.get(priority, (1.0, 0.8, 0.0, alpha))  # Желтый по умолчанию
    
    def export_to_json(self, placements: List[Dict[str, Any]], 
                      output_path: str,
                      constraint_manager: Optional[ConstraintManager] = None,
                      cutting_sequence: Optional[List[int]] = None):
        """
        Экспорт результатов в JSON формат для интеграции с другими системами
        
        :param placements: Список размещенных фигур
        :param output_path: Путь для сохранения JSON файла
        :param constraint_manager: Менеджер технологических ограничений
        :param cutting_sequence: Последовательность резки
        """
        logger.info(f"Экспорт результатов в JSON: {output_path}")
        start_time = time.time()
        
        try:
            # Формирование структуры данных
            export_data = {
                'metadata': {
                    'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'software': 'IAGI Nesting System v1.0',
                    'sheet_size': self.sheet_size,
                    'min_gap': self.min_gap,
                    'standards': self.metadata['standards']
                },
                'placements': [],
                'cutting_sequence': cutting_sequence or list(range(len(placements)))
            }
            
            # Экспорт размещенных фигур
            for i, placement in enumerate(placements):
                shape = placement['shape']
                transformed_shape = shape.apply_transformation(
                    placement['position'], 
                    placement['angle']
                )
                
                placement_data = {
                    'id': i + 1,
                    'name': placement.get('name', f'part_{i+1}'),
                    'position': [float(p) for p in placement['position']],
                    'angle': float(placement['angle']),
                    'priority': placement.get('priority', 1),
                    'area': float(shape.area),
                    'outer_contour': [[float(x), float(y)] for x, y in transformed_shape.outer_contour.tolist()],
                    'inner_contours': [
                        [[float(x), float(y)] for x, y in contour.tolist()]
                        for contour in transformed_shape.inner_contours
                    ],
                    'centroid': [float(c) for c in transformed_shape.centroid.tolist()],
                    'bounding_box': [float(b) for b in transformed_shape.get_bounding_box()]
                }
                export_data['placements'].append(placement_data)
            
            # Экспорт технологических ограничений
            if constraint_manager:
                export_data['constraints'] = {
                    'technology': constraint_manager.current_technology,
                    'min_gap': constraint_manager.current_gap,
                    'orientation_constraints': {
                        k: v for k, v in constraint_manager.orientation_constraints.items()
                    },
                    'placement_priorities': {
                        k: v for k, v in constraint_manager.placement_priorities.items()
                    }
                }
            
            # Сохранение в файл
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elapsed_time = time.time() - start_time
            logger.info(f"JSON экспорт завершен успешно. Файл сохранен: {output_path}")
            logger.info(f"Экспортировано {len(placements)} фигур за {elapsed_time:.2f} секунд")
            
            # Генерация отчета
            self._generate_export_report(placements, output_path, 'JSON')
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте в JSON: {e}")
            raise
    
    def _generate_export_report(self, placements: List[Dict[str, Any]], 
                               output_path: str, format_type: str):
        """Генерация отчета об экспорте"""
        # Расчет метрик
        total_area = sum(placement['shape'].area for placement in placements)
        sheet_area = self.sheet_size[0] * self.sheet_size[1]
        utilization = total_area / sheet_area * 100
        
        # Подсчет количества фигур по приоритетам
        priority_counts = {}
        for placement in placements:
            priority = placement.get('priority', 1)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Генерация отчета
        report = (
            f"=== ОТЧЕТ ОБ ЭКСПОРТЕ ===\n"
            f"Формат: {format_type}\n"
            f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Файл: {output_path}\n\n"
            f"ПАРАМЕТРЫ РАСКРОЯ:\n"
            f"  Размер листа: {self.sheet_size[0]}x{self.sheet_size[1]} мм\n"
            f"  Минимальный зазор: {self.min_gap} мм\n"
            f"  Число деталей: {len(placements)}\n"
            f"  Коэффициент использования материала: {utilization:.2f}%\n\n"
            f"РАСПРЕДЕЛЕНИЕ ПО ПРИОРИТЕТАМ:\n"
        )
        
        for priority, count in sorted(priority_counts.items()):
            report += f"  Приоритет {priority}: {count} деталей\n"
        
        report += f"\nСТАНДАРТЫ: {', '.join(self.metadata['standards'])}"
        
        # Сохранение отчета
        report_path = os.path.splitext(output_path)[0] + '_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Отчет об экспорте сохранен: {report_path}")

def export_results(placements: List[Dict[str, Any]], 
                  output_dir: str,
                  sheet_size: Tuple[float, float],
                  min_gap: float = 1.0,
                  defect_zones: Optional[List[PolygonShape]] = None,
                  constraint_manager: Optional[ConstraintManager] = None,
                  formats: List[str] = ['dxf', 'svg', 'json'],
                  cutting_sequence: Optional[List[int]] = None,
                  create_timestamped_folder: bool = True):
    """
    Функция-обертка для экспорта результатов в несколько форматов
    
    :param placements: Список размещенных фигур
    :param output_dir: Директория для сохранения результатов
    :param sheet_size: Размеры листа (ширина, высота) в мм
    :param min_gap: Минимальный технологический зазор в мм
    :param defect_zones: Список дефектных зон на листе
    :param constraint_manager: Менеджер технологических ограничений
    :param formats: Список форматов для экспорта ['dxf', 'step', 'svg', 'json']
    :param cutting_sequence: Последовательность резки
    :return: Словарь с путями к экспортированным файлам
    """
    if create_timestamped_folder:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join(output_dir, timestamp)
    
    # Создание директории, если она не существует
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            raise OSError(f"Не удалось создать директорию: {output_dir}")
        logger.info(f"Директория для результатов создана: {os.path.abspath(output_dir)}")
    except Exception as e:
        logger.error(f"Ошибка при создании директории {output_dir}: {e}")
        fallback_dir = os.path.join(os.getcwd(), 'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(fallback_dir, exist_ok=True)
        output_dir = fallback_dir
        logger.warning(f"Используется резервная директория: {os.path.abspath(output_dir)}")
    
    # Инициализация экспортера
    exporter = ResultExporter(sheet_size, min_gap)
    
    # Словарь для хранения путей к файлам
    exported_files = {}
    
    # Базовое имя файла
    base_name = f"nesting_result_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Экспорт в указанные форматы
    for fmt in formats:
        try:
            if fmt.lower() == 'dxf':
                output_path = os.path.join(output_dir, f"{base_name}.dxf")
                exporter.export_to_dxf(
                    placements, 
                    output_path, 
                    defect_zones, 
                    constraint_manager
                )
                exported_files['dxf'] = output_path
            
            elif fmt.lower() == 'step':
                if STEP_SUPPORT:
                    output_path = os.path.join(output_dir, f"{base_name}.step")
                    exporter.export_to_step(placements, output_path)
                    exported_files['step'] = output_path
                else:
                    logger.warning("Экспорт в STEP пропущен: библиотека pythonocc-core не установлена")
            
            elif fmt.lower() == 'svg':
                output_path = os.path.join(output_dir, f"{base_name}.svg")
                exporter.export_to_svg(placements, output_path, defect_zones)
                exported_files['svg'] = output_path
            
            elif fmt.lower() == 'json':
                output_path = os.path.join(output_dir, f"{base_name}.json")
                exporter.export_to_json(placements, output_path, constraint_manager, cutting_sequence)
                exported_files['json'] = output_path
            
            else:
                logger.warning(f"Неизвестный формат экспорта: {fmt}")
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в формат {fmt}: {e}")
    
    # Генерация общего отчета
    total_report = (
        f"=== СВОДНЫЙ ОТЧЕТ ОБ ЭКСПОРТЕ ===\n"
        f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Директория: {output_dir}\n\n"
        f"ЭКСПОРТИРОВАННЫЕ ФОРМАТЫ:\n"
    )
    
    for fmt, path in exported_files.items():
        total_report += f"  {fmt.upper()}: {path}\n"
    
    total_report_path = os.path.join(output_dir, f"{base_name}_summary_report.txt")
    with open(total_report_path, 'w', encoding='utf-8') as f:
        f.write(total_report)
    
    logger.info(f"Сводный отчет об экспорте сохранен: {total_report_path}")
    
    return exported_files