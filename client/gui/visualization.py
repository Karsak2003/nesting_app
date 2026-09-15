#region Imports
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
#endregion

#region Class Definition
class RealTimeVisualization(FigureCanvas):
    """
    Виджет для визуализации процесса раскроя ИАГИ в реальном времени.
    
    Особенности:
    - Атомарное обновление через update_from_status() (без мигания)
    - Хранение патчей в словаре для точечного обновления позиций
    - Обработка is_frozen (полупрозрачность для замороженных фигур)
    - Нормированные векторы сил/скоростей
    - Дефектные зоны с штриховкой
    """
    
    def __init__(self, parent=None, sheet_size=(2000, 1000), width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Геометрия листа
        self.sheet_size = sheet_size
        
        # Состояние визуализации
        self.placed_shapes = []
        self.defect_zones = []
        self.forces = {}
        
        # Флаги отображения
        self.show_forces = False
        self.show_velocities = False
        self.show_sdf = False
        
        # Хранилища артистов для точечного обновления (ключ — id фигуры)
        self.shape_patches = {}
        self.shape_labels = {}
        self.shape_centroids = {}
        self.defect_patches = {}
        self.force_artists = None
        self.velocity_artists = None
        self.sheet_rect = None
        self.utilization_text = None
        self.legend = None
        
        # Параметры нормирования векторов
        self.force_scale = 0.5      # масштаб стрелок сил
        self.velocity_scale = 2.0   # масштаб стрелок скоростей
        
        self._setup_plot()
        
#region Plot Setup
    def _setup_plot(self):
        """Инициализация графика (вызывается один раз при создании)"""
        self.axes.clear()
        self.axes.set_xlim(-50, self.sheet_size[0] + 50)
        self.axes.set_ylim(-50, self.sheet_size[1] + 50)
        self.axes.set_aspect('equal')
        self.axes.set_title('Раскрой ИАГИ — реальное время', fontsize=13, fontweight='bold')
        self.axes.set_xlabel('X (мм)', fontsize=11)
        self.axes.set_ylabel('Y (мм)', fontsize=11)
        self.axes.grid(True, linestyle='--', alpha=0.5, zorder=0)
        
        # Рамка листа (создаётся один раз)
        self.sheet_rect = Rectangle(
            (0, 0), self.sheet_size[0], self.sheet_size[1],
            fill=False, edgecolor='black', linewidth=2.5, linestyle='-', zorder=1
        )
        self.axes.add_patch(self.sheet_rect)
        
        # Подписи размеров
        self.axes.text(
            self.sheet_size[0] / 2, -30,
            f"{int(self.sheet_size[0])} мм",
            ha='center', va='top', fontsize=10, color='#555'
        )
        self.axes.text(
            -30, self.sheet_size[1] / 2,
            f"{int(self.sheet_size[1])} мм",
            ha='right', va='center', rotation=90, fontsize=10, color='#555'
        )
        
        # Заглушка до получения данных
        self.waiting_text = self.axes.text(
            0.5, 0.5, "Ожидание данных от сервера...",
            transform=self.axes.transAxes, ha='center', va='center',
            fontsize=14, color='gray', style='italic'
        )
        
        # Текст утилизации (правый верхний угол)
        self.utilization_text = self.axes.text(
            0.98, 0.98, "η: —",
            transform=self.axes.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8)
        )
        
        self.draw_idle()
#endregion

#region Core Update
    def update_from_status(self, status: dict):
        """
        Атомарное обновление визуализации из статуса сервера.
        
        Args:
            status: полный ответ от GET /api/status/{task_id}
                - current_shapes: List[Dict] с полями id, name, contour, centroid,
                  angle, priority, is_frozen, velocity, angular_velocity
                - utilization: float (коэффициент использования, %)
                - defect_zones: List[Dict] (опционально)
        """
        shapes_data = status.get('current_shapes') or []
        print(f"{status.get('current_shapes')}||{shapes_data}")
        utilization = status.get('utilization', 0.0) or 0.0
        print(f"{status.get('utilization', 0.0)}||{utilization}")
        
        
        # Скрываем заглушку при первом получении данных
        if shapes_data and hasattr(self, 'waiting_text') and self.waiting_text:
            self.waiting_text.set_visible(False)
            self.waiting_text = None
        
        # 1. Обновление/создание фигур
        current_ids = set()
        for shape in shapes_data:
            shape_id = shape.get('id')
            if shape_id is None:
                continue
            current_ids.add(shape_id)
            self._update_single_shape(shape)
        
        # 2. Удаление фигур, которых больше нет в статусе
        removed_ids = set(self.shape_patches.keys()) - current_ids
        for shape_id in removed_ids:
            self._remove_shape(shape_id)
        
        # 3. Обновление дефектных зон
        defect_zones = status.get('defect_zones', [])
        if defect_zones:
            self._update_defect_zones(defect_zones)
        
        # 4. Обновление векторов сил/скоростей
        if self.show_velocities:
            self._update_velocity_arrows(shapes_data)
        elif self.velocity_artists is not None:
            self.velocity_artists.remove()
            self.velocity_artists = None
        
        # 5. Обновление текста утилизации
        self.utilization_text.set_text(f"η: {utilization:.1f}%")
        
        # 6. Перерисовка БЕЗ МИГАНИЯ
        self.draw_idle()
    
    def _update_single_shape(self, shape: dict):
        """Обновление или создание патча одной фигуры"""
        shape_id = shape.get('id')
        contour = shape.get('contour', [])
        if len(contour) < 3:
            return
        
        priority = shape.get('priority', 2)
        is_frozen = shape.get('is_frozen', False)
        color = self._get_priority_color(priority)
        alpha = 0.35 if is_frozen else 0.85
        
        if shape_id in self.shape_patches:
            # ТОЧЕЧНОЕ ОБНОВЛЕНИЕ — без пересоздания патча
            patch = self.shape_patches[shape_id]
            patch.set_xy(contour)
            patch.set_alpha(alpha)
            patch.set_facecolor(color[:3])
            
            # Обновление центроида и подписи
            if 'centroid' in shape and shape_id in self.shape_centroids:
                cx, cy = shape['centroid']
                self.shape_centroids[shape_id].set_data([cx], [cy])
            if shape_id in self.shape_labels:
                label = self.shape_labels[shape_id]
                if 'centroid' in shape:
                    cx, cy = shape['centroid']
                    label.set_position((cx, cy))
                # Обновление текста (с учётом frozen)
                new_text = f"{shape.get('name', f'P{priority}')}\nP={priority}"
                if is_frozen:
                    new_text += "\n🔒"
                label.set_text(new_text)
        else:
            # СОЗДАНИЕ НОВОГО ПАТЧА
            self._create_shape_patch(shape, priority, is_frozen, color, alpha)
    
    def _create_shape_patch(self, shape: dict, priority: int, is_frozen: bool,
                            color: tuple, alpha: float):
        """Создание нового патча для фигуры"""
        shape_id = shape.get('id')
        contour = shape.get('contour', [])
        
        # Основной полигон
        polygon = MplPolygon(
            contour, closed=True, fill=True,
            facecolor=color[:3], edgecolor='black',
            linewidth=1.5 if not is_frozen else 1.0,
            linestyle='-' if not is_frozen else '--',
            zorder=10, alpha=alpha
        )
        self.axes.add_patch(polygon)
        self.shape_patches[shape_id] = polygon
        
        # Центроид (точка)
        if 'centroid' in shape:
            cx, cy = shape['centroid']
            centroid_pt, = self.axes.plot(
                [cx], [cy], 'ko', markersize=4, zorder=11
            )
            self.shape_centroids[shape_id] = centroid_pt
            
            # Подпись
            label_text = f"{shape.get('name', f'P{priority}')}\nP={priority}"
            if is_frozen:
                label_text += "\n🔒"
            label = self.axes.text(
                cx, cy, label_text,
                ha='center', va='center', fontsize=8,
                bbox=dict(
                    facecolor='white', alpha=0.75, edgecolor='none',
                    boxstyle='round,pad=0.3'
                ),
                zorder=12
            )
            self.shape_labels[shape_id] = label
    
    def _remove_shape(self, shape_id):
        """Удаление патча фигуры со всеми артефактами"""
        if shape_id in self.shape_patches:
            self.shape_patches[shape_id].remove()
            del self.shape_patches[shape_id]
        if shape_id in self.shape_centroids:
            self.shape_centroids[shape_id].remove()
            del self.shape_centroids[shape_id]
        if shape_id in self.shape_labels:
            self.shape_labels[shape_id].remove()
            del self.shape_labels[shape_id]
#endregion

#region Defect Zones
    def _update_defect_zones(self, defect_zones: list):
        """Обновление дефектных зон (штриховка + красная обводка)"""
        # Удаляем старые
        for patch in self.defect_patches.values():
            patch.remove()
        self.defect_patches.clear()
        
        for i, defect in enumerate(defect_zones):
            contour = defect.get('contour', [])
            if len(contour) < 3:
                continue
            
            defect_polygon = MplPolygon(
                contour, closed=True, fill=True,
                facecolor='red', alpha=0.25,
                edgecolor='darkred', linewidth=1.5,
                linestyle='--', hatch='///', zorder=5
            )
            self.axes.add_patch(defect_polygon)
            self.defect_patches[i] = defect_polygon
            
            if 'centroid' in defect:
                cx, cy = defect['centroid']
                self.axes.text(
                    cx, cy, f"ДЕФЕКТ {i+1}",
                    ha='center', va='center', fontsize=8, color='darkred',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6
                )
#endregion

#region Velocity & Force Vectors
    def _update_velocity_arrows(self, shapes_data: list):
        """Отрисовка векторов скоростей (нормированных)"""
        # Удаляем старые стрелки
        if self.velocity_artists is not None:
            self.velocity_artists.remove()
        
        segments = []
        for shape in shapes_data:
            if shape.get('is_frozen', False):
                continue
            velocity = shape.get('velocity', [0, 0])
            if velocity is None or len(velocity) < 2:
                continue
            v = np.array(velocity, dtype=float)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                continue
            
            centroid = shape.get('centroid', [0, 0])
            start = np.array(centroid, dtype=float)
            # Нормирование + масштаб
            direction = v / v_norm
            end = start + direction * self.velocity_scale * min(v_norm * 10, 50)
            segments.append([start, end])
        
        if segments:
            lc = LineCollection(
                segments, colors='green', linewidths=1.5,
                zorder=15, alpha=0.8
            )
            lc.set_capstyle('round')
            self.axes.add_collection(lc)
            self.velocity_artists = lc
        else:
            self.velocity_artists = None
#endregion

#region Helpers
    def _get_priority_color(self, priority: int, alpha: float = 0.85) -> tuple:
        """Цветовая схема по приоритету (согласовано с темой)"""
        color_map = {
            1: (0.20, 0.70, 0.30, alpha),  # зелёный — высокий
            2: (0.20, 0.45, 0.85, alpha),  # синий — средний
            3: (0.95, 0.55, 0.15, alpha),  # оранжевый — низкий
            4: (0.70, 0.25, 0.75, alpha),  # фиолетовый
            5: (0.50, 0.50, 0.50, alpha),  # серый
        }
        return color_map.get(priority, (0.30, 0.60, 0.90, alpha))
#endregion

#region Toggles & Setters
    def toggle_forces(self, show: bool = True):
        self.show_forces = show
        self.draw_idle()
    
    def toggle_velocities(self, show: bool = True):
        self.show_velocities = show
        if not show and self.velocity_artists is not None:
            self.velocity_artists.remove()
            self.velocity_artists = None
        self.draw_idle()
    
    def toggle_sdf(self, show: bool = True):
        self.show_sdf = show
        self.draw_idle()
    
    def set_defect_zones(self, defect_zones: list):
        self.defect_zones = defect_zones
        self._update_defect_zones(defect_zones)
        self.draw_idle()
    
    def set_sheet_size(self, width: float, height: float):
        """Изменение размера листа (пересоздаёт график)"""
        self.sheet_size = (width, height)
        self._setup_plot()
    
    def reset(self):
        """Полный сброс визуализации"""
        for patch in list(self.shape_patches.values()):
            patch.remove()
        self.shape_patches.clear()
        
        for pt in list(self.shape_centroids.values()):
            pt.remove()
        self.shape_centroids.clear()
        
        for label in list(self.shape_labels.values()):
            label.remove()
        self.shape_labels.clear()
        
        for patch in list(self.defect_patches.values()):
            patch.remove()
        self.defect_patches.clear()
        
        if self.velocity_artists is not None:
            self.velocity_artists.remove()
            self.velocity_artists = None
        
        self.utilization_text.set_text("η: —")
        
        # Возвращаем заглушку
        self.waiting_text = self.axes.text(
            0.5, 0.5, "Ожидание данных от сервера...",
            transform=self.axes.transAxes, ha='center', va='center',
            fontsize=14, color='gray', style='italic'
        )
        
        self.draw_idle()
#endregion

#region Backward Compatibility
    def update_visualization(self, shapes_data: list, utilization: float = 0.0):
        """
        Обратная совместимость: обновление визуализации из финального результата.
        
        Используется в refresh_visualization() после завершения задачи,
        когда сервер возвращает 'result' (финальное размещение) вместо 'current_shapes'.
        
        Конвертирует формат 'result' в формат 'current_shapes' и вызывает
        update_from_status().
        """
        if not shapes_data:
            self.reset()
            return
        
        # Конвертация формата result → current_shapes
        converted_shapes = []
        for i, shape in enumerate(shapes_data):
            # Извлекаем контур (может быть Shapely Polygon или список координат)
            contour = shape.get('contour', [])
            if not contour and 'shape' in shape:
                # Если это объект с атрибутом polygon (Shapely)
                shapely_shape = shape['shape']
                if hasattr(shapely_shape, 'polygon'):
                    contour = list(shapely_shape.polygon.exterior.coords)[:-1]
                elif hasattr(shapely_shape, 'exterior'):
                    contour = list(shapely_shape.exterior.coords)[:-1]
            
            # Центроид
            centroid = shape.get('centroid')
            if centroid is None and 'position' in shape:
                centroid = list(shape['position'])
            
            converted_shapes.append({
                'id': i,
                'name': shape.get('name', f'Part_{i+1}'),
                'contour': [[round(x, 3), round(y, 3)] for x, y in contour],
                'centroid': centroid,
                'angle': shape.get('angle', 0.0),
                'priority': shape.get('priority', 2),
                'is_frozen': shape.get('is_frozen', True),  # финальные — заморожены
                'velocity': [0.0, 0.0],
                'angular_velocity': 0.0,
            })
        
        # Формируем статус и вызываем основной метод
        status = {
            'current_shapes': converted_shapes,
            'utilization': utilization,
        }
        self.update_from_status(status)
#endregion
    


#region Export
    def export_to_svg(self, filename: str):
        self.fig.savefig(filename, format='svg', bbox_inches='tight')
    
    def export_to_png(self, filename: str):
        self.fig.savefig(filename, format='png', dpi=150, bbox_inches='tight')
#endregion

#endregion