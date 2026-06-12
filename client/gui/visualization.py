#region Imports
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.lines import Line2D
#endregion

class RealTimeVisualization(FigureCanvas):
    """Виджет для визуализации процесса раскроя в реальном времени"""
    def __init__(self, parent=None, sheet_size=(2000, 1000), width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.sheet_size = sheet_size
        self.placed_shapes = []
        self.defect_zones = []
        self.forces = {}
        self.show_forces = False
        self.show_velocities = False
        self.show_sdf = False
        self.setup_plot()

    #region Plot Setup & Drawing
    def setup_plot(self):
        self.axes.clear()
        self.axes.set_xlim(0, self.sheet_size[0])
        self.axes.set_ylim(0, self.sheet_size[1])
        self.axes.set_aspect('equal')
        self.axes.set_title('Результат раскроя ИАГИ', fontsize=14)
        self.axes.set_xlabel('X (мм)', fontsize=12)
        self.axes.set_ylabel('Y (мм)', fontsize=12)
        self.axes.grid(True, linestyle='--', alpha=0.7) 
        
        sheet_rect = Rectangle((0, 0), self.sheet_size[0], self.sheet_size[1],
                              fill=False, edgecolor='black', linewidth=2, linestyle='-')
        self.axes.add_patch(sheet_rect)
        
        self.axes.text(self.sheet_size[0]/2, -20, f"{self.sheet_size[0]} мм", ha='center', va='top', fontsize=10)
        self.axes.text(-20, self.sheet_size[1]/2, f"{self.sheet_size[1]} мм", ha='right', va='center', rotation=90, fontsize=10)
        self.draw()

    def update_visualization(self, shapes_data: list, utilization: float = 0.0):
        self.axes.clear()
        self.setup_plot()
        
        # Защита: если данные не пришли или пусты, просто рисуем пустой лист с утилизацией 0
        if not shapes_data:
            self.axes.text(0.5, 0.5, "Ожидание данных от сервера...", 
                        transform=self.axes.transAxes, ha='center', va='center', fontsize=14, color='gray')
            self.draw()
            return
        
        self.placed_shapes = shapes_data
        self._draw_defect_zones()
        
        max_priority = max((shape.get('priority', 2) for shape in shapes_data), default=1)
        for priority in range(max_priority, 0, -1):
            for shape in shapes_data:
                if shape.get('priority', 2) == priority:
                    self._draw_shape(shape, priority)
        
        if self.show_forces: self._draw_forces()
        if self.show_velocities: self._draw_velocities()
        self._update_legend()
        
        self.axes.text(0.02, 0.98, f"Использование: {utilization:.1f}%", transform=self.axes.transAxes,
                      fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Явный вызов перерисовки холста
        self.draw()

    def _draw_shape(self, shape: dict, priority: int):
        contour = shape.get('contour', [])
        if len(contour) < 3: return
        color = self._get_priority_color(priority)
        polygon = MplPolygon(contour, closed=True, fill=True, facecolor=color[:3],
                           edgecolor='black', linewidth=1.5, zorder=10, alpha=color[3] if len(color) > 3 else 0.8)
        self.axes.add_patch(polygon)
        if 'centroid' in shape:
            cx, cy = shape['centroid']
            self.axes.plot(cx, cy, 'ko', markersize=4, zorder=11)
            self.axes.text(cx, cy, f"{shape.get('name', f'Part_{priority}')}\nP={priority}", 
                          ha='center', va='center', fontsize=8,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'), zorder=12)

    def _draw_defect_zones(self):
        for i, defect in enumerate(self.defect_zones):
            contour = defect.get('contour', [])
            if len(contour) < 3: continue
            defect_polygon = MplPolygon(contour, closed=True, fill=True, facecolor='red', alpha=0.3,
                                      edgecolor='darkred', linewidth=1.5, linestyle='--', zorder=5)
            self.axes.add_patch(defect_polygon)
            if 'centroid' in defect:
                cx, cy = defect['centroid']
                self.axes.text(cx, cy, f"ДЕФЕКТ {i+1}", ha='center', va='center', fontsize=8, color='darkred',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=6)

    def _draw_forces(self):
        for shape_id, force_data in self.forces.items():
            if isinstance(force_data, dict) and 'vector' in force_data:
                force = np.array(force_data['vector'])
                position = force_data.get('position', [0, 0])
                self.axes.annotate('', xy=position + force * 0.5, xytext=position,
                                 arrowprops=dict(arrowstyle='->', color='blue', lw=2), zorder=15)

    def _draw_velocities(self):
        for shape in self.placed_shapes:
            if 'velocity' in shape and not shape.get('is_frozen', False):
                velocity = np.array(shape['velocity'])
                position = shape.get('centroid', [0, 0])
                self.axes.annotate('', xy=position + velocity * 2.0, xytext=position,
                                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5), zorder=15)

    def _get_priority_color(self, priority, alpha=0.8):
        color_map = {1: (0.0, 0.8, 0.0, alpha), 2: (0.0, 0.0, 1.0, alpha), 3: (1.0, 0.5, 0.0, alpha),
                     4: (0.8, 0.0, 0.8, alpha), 5: (0.5, 0.5, 0.5, alpha)}
        return color_map.get(priority, (0.2, 0.6, 1.0, alpha))

    def _update_legend(self):
        legend_elements = [
            Line2D([0], [0], color=self._get_priority_color(1), lw=4, label='Приоритет 1 (высокий)'),
            Line2D([0], [0], color=self._get_priority_color(2), lw=4, label='Приоритет 2 (средний)'),
            Line2D([0], [0], color=self._get_priority_color(3), lw=4, label='Приоритет 3 (низкий)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Дефектные зоны'),
            Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Силы'),
            Line2D([0], [0], color='green', lw=2, linestyle='-', label='Скорости')
        ]
        self.axes.legend(handles=legend_elements, loc='upper right', fontsize=9)
    #endregion

    #region Toggles & Setters
    def toggle_forces(self, show=True): self.show_forces = show; self.update_visualization(self.placed_shapes)
    def toggle_velocities(self, show=True): self.show_velocities = show; self.update_visualization(self.placed_shapes)
    def toggle_sdf(self, show=True): self.show_sdf = show; self.update_visualization(self.placed_shapes)
    def set_defect_zones(self, defect_zones: list): self.defect_zones = defect_zones; self.update_visualization(self.placed_shapes)
    def set_forces(self, forces: dict): self.forces = forces; self.update_visualization(self.placed_shapes) if self.show_forces else None
    def set_sheet_size(self, width: float, height: float): self.sheet_size = (width, height); self.setup_plot()
    #endregion

    #region Export
    def export_to_svg(self, filename: str): self.fig.savefig(filename, format='svg', bbox_inches='tight')
    def export_to_png(self, filename: str): self.fig.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    #endregion