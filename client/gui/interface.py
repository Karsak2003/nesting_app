"""
Графический интерфейс клиентской части ИАГИ
Объединяет функционал из gui/interface.py и client/main.py
Предназначен для работы с сервером FastAPI через HTTP API
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QTabWidget, QFileDialog, QProgressBar, QMessageBox,
                             QTextEdit, QSplitter, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QDialog, QFormLayout, QDialogButtonBox, QLineEdit,
                             QRadioButton, QButtonGroup, QMenu, QAction, QStatusBar,
                             QListWidget, QListWidgetItem, QFrame)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QSettings, QProcess
from PyQt5.QtGui import QColor, QPalette, QFont, QPixmap, QPainter


# Загрузка настроек клиента
def get_client_config() -> Dict[str, Any]:
    """Загрузка конфигурации клиента из JSON файла"""
    config_path = Path(__file__).parent.parent / "config" / "settings.json"
    default_config = {
        "server": {
            "url": "http://localhost:8000",
            "timeout": 300
        },
        "theme": "system",
        "local_server": {
            "enabled": True,
            "auto_start": False,
            "port": 8000,
            "show_console": True
        },
        "last_used": {
            "profile": "medium_precision",
            "algorithm": "sequential",
            "technology": "laser"
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Объединяем с дефолтными значениями
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                    elif isinstance(default_config[key], dict):
                        for subkey in default_config[key]:
                            if subkey not in config[key]:
                                config[key][subkey] = default_config[key][subkey]
                return config
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}")
    
    return default_config


def save_client_config(config: Dict[str, Any]):
    """Сохранение конфигурации клиента в JSON файл"""
    config_path = Path(__file__).parent.parent / "config" / "settings.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


class RealTimeVisualization(FigureCanvas):
    """
    Виджет для визуализации процесса раскроя в реальном времени
    Получает данные от сервера и отображает результат оптимизации
    """
    
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
        self.color_map = {}
        self.setup_plot()
    
    def setup_plot(self):
        """Настройка начального состояния графика"""
        self.axes.clear()
        self.axes.set_xlim(0, self.sheet_size[0])
        self.axes.set_ylim(0, self.sheet_size[1])
        self.axes.set_aspect('equal')
        self.axes.set_title('Результат раскроя ИАГИ', fontsize=14)
        self.axes.set_xlabel('X (мм)', fontsize=12)
        self.axes.set_ylabel('Y (мм)', fontsize=12)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Отрисовка границ листа
        sheet_rect = Rectangle((0, 0), self.sheet_size[0], self.sheet_size[1],
                              fill=False, edgecolor='black', linewidth=2, linestyle='-')
        self.axes.add_patch(sheet_rect)
        
        # Добавление подписи размеров листа
        self.axes.text(self.sheet_size[0]/2, -20, f"{self.sheet_size[0]} мм", 
                      ha='center', va='top', fontsize=10)
        self.axes.text(-20, self.sheet_size[1]/2, f"{self.sheet_size[1]} мм", 
                      ha='right', va='center', rotation=90, fontsize=10)
        
        self.draw()
    
    def update_visualization(self, shapes_data: list, utilization: float = 0.0):
        """
        Обновление визуализации с данными от сервера
        
        :param shapes_data: Список словарей с данными о размещённых фигурах
        :param utilization: Коэффициент использования материала (%)
        """
        self.axes.clear()
        self.setup_plot()
        
        self.placed_shapes = shapes_data
        
        # Отрисовка дефектных зон
        self._draw_defect_zones()
        
        # Отрисовка размещённых фигур по приоритетам (для корректного наложения)
        max_priority = max(shape.get('priority', 2) for shape in shapes_data) if shapes_data else 1
        
        for priority in range(max_priority, 0, -1):
            for shape in shapes_data:
                if shape.get('priority', 2) == priority:
                    self._draw_shape(shape, priority)
        
        # Отрисовка сил и скоростей, если включено
        if self.show_forces:
            self._draw_forces()
        
        if self.show_velocities:
            self._draw_velocities()
        
        # Обновление легенды
        self._update_legend()
        
        # Добавление информации об утилизации
        self.axes.text(0.02, 0.98, f"Использование: {utilization:.1f}%",
                      transform=self.axes.transAxes, fontsize=12,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.draw()
    
    def _draw_shape(self, shape: dict, priority: int):
        """Отрисовка отдельной фигуры"""
        contour = shape.get('contour', [])
        if len(contour) < 3:
            return
            
        color = self._get_priority_color(priority)
        
        # Основной контур фигуры
        polygon = MplPolygon(contour, closed=True,
                           fill=True, facecolor=color[:3],
                           edgecolor='black', linewidth=1.5, zorder=10,
                           alpha=color[3] if len(color) > 3 else 0.8)
        self.axes.add_patch(polygon)
        
        # Центр масс
        if 'centroid' in shape:
            cx, cy = shape['centroid']
            self.axes.plot(cx, cy, 'ko', markersize=4, zorder=11)
        
        # Подпись с именем и приоритетом
        name = shape.get('name', f'Part_{priority}')
        if 'centroid' in shape:
            self.axes.text(cx, cy, f'{name}\nP={priority}', 
                          ha='center', va='center', fontsize=8,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                          zorder=12)
    
    def _draw_defect_zones(self):
        """Отрисовка дефектных зон"""
        for i, defect in enumerate(self.defect_zones):
            contour = defect.get('contour', [])
            if len(contour) < 3:
                continue
                
            defect_polygon = MplPolygon(contour, closed=True,
                                      fill=True, facecolor='red', alpha=0.3,
                                      edgecolor='darkred', linewidth=1.5,
                                      linestyle='--', zorder=5)
            self.axes.add_patch(defect_polygon)
            
            # Текстовая метка для дефекта
            if 'centroid' in defect:
                centroid = defect['centroid']
                self.axes.text(centroid[0], centroid[1], f"ДЕФЕКТ {i+1}", 
                              ha='center', va='center', fontsize=8, color='darkred',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                              zorder=6)
    
    def _draw_forces(self):
        """Отрисовка векторов сил"""
        scale = 0.5  # Масштаб для отображения сил
        
        for shape_id, force_data in self.forces.items():
            if isinstance(force_data, dict) and 'vector' in force_data:
                force = np.array(force_data['vector'])
                position = force_data.get('position', [0, 0])
                
                # Вектор силы
                end_point = position + force * scale
                self.axes.annotate('', xy=end_point, xytext=position,
                                 arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                                 zorder=15)
    
    def _draw_velocities(self):
        """Отрисовка векторов скоростей"""
        scale = 2.0  # Масштаб для отображения скоростей
        
        for shape in self.placed_shapes:
            if 'velocity' in shape and not shape.get('is_frozen', False):
                velocity = np.array(shape['velocity'])
                position = shape.get('centroid', [0, 0])
                
                # Вектор скорости
                end_point = position + velocity * scale
                self.axes.annotate('', xy=end_point, xytext=position,
                                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                                 zorder=15)
    
    def _get_priority_color(self, priority, alpha=0.8):
        """Получение цвета для заданного приоритета"""
        color_map = {
            1: (0.0, 0.8, 0.0, alpha),   # Зеленый для высокого приоритета
            2: (0.0, 0.0, 1.0, alpha),   # Синий для среднего приоритета
            3: (1.0, 0.5, 0.0, alpha),   # Оранжевый для низкого приоритета
            4: (0.8, 0.0, 0.8, alpha),   # Фиолетовый
            5: (0.5, 0.5, 0.5, alpha)    # Серый для остальных
        }
        return color_map.get(priority, (0.2, 0.6, 1.0, alpha))  # Голубой по умолчанию
    
    def _update_legend(self):
        """Обновление легенды графика"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], color=self._get_priority_color(1), lw=4, label='Приоритет 1 (высокий)'),
            Line2D([0], [0], color=self._get_priority_color(2), lw=4, label='Приоритет 2 (средний)'),
            Line2D([0], [0], color=self._get_priority_color(3), lw=4, label='Приоритет 3 (низкий)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
                  label='Дефектные зоны'),
            Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Силы (при включении)'),
            Line2D([0], [0], color='green', lw=2, linestyle='-', label='Скорости (при включении)')
        ]
        
        self.axes.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    def toggle_forces(self, show=True):
        """Переключение отображения сил"""
        self.show_forces = show
        self.update_visualization(self.placed_shapes)
    
    def toggle_velocities(self, show=True):
        """Переключение отображения скоростей"""
        self.show_velocities = show
        self.update_visualization(self.placed_shapes)
    
    def toggle_sdf(self, show=True):
        """Переключение отображения полей расстояний"""
        self.show_sdf = show
        self.update_visualization(self.placed_shapes)
    
    def set_defect_zones(self, defect_zones: list):
        """Установка дефектных зон для отображения"""
        self.defect_zones = defect_zones
        self.update_visualization(self.placed_shapes)
    
    def set_forces(self, forces: dict):
        """Установка данных о силах для отображения"""
        self.forces = forces
        if self.show_forces:
            self.update_visualization(self.placed_shapes)
    
    def set_sheet_size(self, width: float, height: float):
        """Изменение размеров листа"""
        self.sheet_size = (width, height)
        self.setup_plot()
    
    def export_to_svg(self, filename: str):
        """Экспорт текущей визуализации в SVG"""
        self.fig.savefig(filename, format='svg', bbox_inches='tight')
    
    def export_to_png(self, filename: str):
        """Экспорт текущей визуализации в PNG"""
        self.fig.savefig(filename, format='png', dpi=150, bbox_inches='tight')


class ServerConnectionDialog(QDialog):
    """Диалог настройки подключения к серверу"""
    
    def __init__(self, parent=None, current_url: str = "http://localhost:8000", current_theme: str = "system"):
        super().__init__(parent)
        self.setWindowTitle("Настройка подключения к серверу")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.current_url = current_url
        self.current_theme = current_theme
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Группа настроек подключения
        conn_group = QGroupBox("Параметры подключения")
        conn_layout = QFormLayout()
        
        # Адрес сервера
        self.server_edit = QLineEdit()
        self.server_edit.setText(self.current_url)
        self.server_edit.setPlaceholderText("http://localhost:8000")
        conn_layout.addRow("Адрес сервера:", self.server_edit)
        
        # Быстрые пресеты
        preset_layout = QHBoxLayout()
        self.local_radio = QRadioButton("Локальный (localhost:8000)")
        self.local_radio.setChecked(True)
        self.remote_radio = QRadioButton("Удаленный")
        
        button_group = QButtonGroup(self)
        button_group.addButton(self.local_radio)
        button_group.addButton(self.remote_radio)
        
        self.local_radio.toggled.connect(lambda checked: self.server_edit.setText(
            "http://localhost:8000" if checked else self.server_edit.text()
        ))
        
        preset_layout.addWidget(self.local_radio)
        preset_layout.addWidget(self.remote_radio)
        conn_layout.addRow("Пресет:", preset_layout)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # Группа настроек темы
        theme_group = QGroupBox("Цветовая тема")
        theme_layout = QVBoxLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Системная", "system")
        self.theme_combo.addItem("Светлая", "light")
        self.theme_combo.addItem("Тёмная", "dark")
        
        # Устанавливаем текущую тему
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == self.current_theme:
                self.theme_combo.setCurrentIndex(i)
                break
        
        theme_layout.addWidget(QLabel("Выберите тему:"))
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def get_server_url(self) -> str:
        return self.server_edit.text().rstrip('/')
    
    def get_theme(self) -> str:
        return self.theme_combo.currentData()


class ControlPanel(QWidget):
    """
    Панель управления процессом раскроя ИАГИ
    Согласно разделу 3.3.4 и 3.3.6, пользователь должен иметь возможность гибко
    настраивать параметры динамической системы для достижения оптимального результата.
    """
    
    start_optimization = pyqtSignal()
    stop_optimization = pyqtSignal()
    pause_optimization = pyqtSignal()
    reset_simulation = pyqtSignal()
    export_results = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_profile = 'medium_precision'
        self.placement_mode = 'sequential'
        self.technology = 'laser'
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Профиль конфигурации
        profile_group = QGroupBox("Профиль конфигурации")
        profile_layout = QVBoxLayout()
        
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(['high_precision', 'medium_precision', 'low_precision'])
        self.profile_combo.setCurrentText(self.current_profile)
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)
        profile_layout.addWidget(QLabel("Выберите профиль для отрасли:"))
        profile_layout.addWidget(self.profile_combo)
        
        profile_info = QTextEdit()
        profile_info.setReadOnly(True)
        profile_info.setFixedHeight(80)
        profile_info.setText(
            "high_precision: Авиация, космос (точность 0.05 мм)\n"
            "medium_precision: Машиностроение (точность 0.1 мм)\n"
            "low_precision: Текстиль, картон (точность 0.5 мм)"
        )
        profile_layout.addWidget(profile_info)
        
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)
        
        # Режим размещения
        mode_group = QGroupBox("Режим размещения")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['sequential', 'parallel', 'hybrid'])
        self.mode_combo.setCurrentText(self.placement_mode)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("Выберите режим размещения:"))
        mode_layout.addWidget(self.mode_combo)
        
        mode_info = QTextEdit()
        mode_info.setReadOnly(True)
        mode_info.setFixedHeight(80)
        mode_info.setText(
            "sequential: Технологически корректный раскрой\n"
            "parallel: Максимизация плотности упаковки\n"
            "hybrid: Комбинация методов (ГА-ИАГИ, PSO-ИАГИ)"
        )
        mode_layout.addWidget(mode_info)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Технологические параметры
        tech_group = QGroupBox("Технологические параметры")
        tech_layout = QVBoxLayout()
        
        tech_layout.addWidget(QLabel("Технология резки:"))
        self.tech_combo = QComboBox()
        self.tech_combo.addItems(['laser', 'plasma', 'waterjet', 'default'])
        self.tech_combo.setCurrentText(self.technology)
        self.tech_combo.currentTextChanged.connect(self.on_technology_changed)
        tech_layout.addWidget(self.tech_combo)
        
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Минимальный зазор (мм):"))
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.1, 10.0)
        self.gap_spin.setValue(0.5)
        self.gap_spin.setSingleStep(0.1)
        gap_layout.addWidget(self.gap_spin)
        tech_layout.addLayout(gap_layout)
        
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Макс. время (сек):"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(10, 3600)
        self.time_spin.setValue(300)
        self.time_spin.setSingleStep(30)
        time_layout.addWidget(self.time_spin)
        tech_layout.addLayout(time_layout)
        
        tech_group.setLayout(tech_layout)
        layout.addWidget(tech_group)
        
        # Размеры листа
        sheet_group = QGroupBox("Размеры листа")
        sheet_layout = QVBoxLayout()
        
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Ширина (мм):"))
        self.sheet_width_spin = QDoubleSpinBox()
        self.sheet_width_spin.setRange(100, 10000)
        self.sheet_width_spin.setValue(2000)
        self.sheet_width_spin.setSingleStep(100)
        width_layout.addWidget(self.sheet_width_spin)
        sheet_layout.addLayout(width_layout)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Высота (мм):"))
        self.sheet_height_spin = QDoubleSpinBox()
        self.sheet_height_spin.setRange(100, 10000)
        self.sheet_height_spin.setValue(1000)
        self.sheet_height_spin.setSingleStep(100)
        height_layout.addWidget(self.sheet_height_spin)
        sheet_layout.addLayout(height_layout)
        
        sheet_group.setLayout(sheet_layout)
        layout.addWidget(sheet_group)
        
        # Кнопки управления
        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Запустить")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Остановить")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        control_layout.addLayout(button_layout)
        
        # Индикатор прогресса
        self.progress_label = QLabel("Готов к работе")
        control_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        # Статистика
        stats_layout = QHBoxLayout()
        self.utilization_label = QLabel("Использование: -")
        stats_layout.addWidget(self.utilization_label)
        control_layout.addLayout(stats_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Экспорт
        export_group = QGroupBox("Экспорт результатов")
        export_layout = QHBoxLayout()
        
        self.dxf_button = QPushButton("DXF")
        self.dxf_button.setEnabled(False)
        export_layout.addWidget(self.dxf_button)
        
        self.svg_button = QPushButton("SVG")
        self.svg_button.setEnabled(False)
        export_layout.addWidget(self.svg_button)
        
        self.json_button = QPushButton("JSON")
        self.json_button.setEnabled(False)
        export_layout.addWidget(self.json_button)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        self.setLayout(layout)
    
    def on_profile_changed(self, profile_name):
        """Обработчик изменения профиля"""
        self.current_profile = profile_name
        QMessageBox.information(self, "Профиль изменен", 
                               f"Применен профиль: {profile_name}")
    
    def on_mode_changed(self, mode):
        """Обработчик изменения режима"""
        self.placement_mode = mode
        
        if mode == 'hybrid':
            QMessageBox.information(self, "Гибридный режим", 
                                   "Включены гибридные алгоритмы (ГА-ИАГИ, PSO-ИАГИ).\n"
                                   "Оптимизация может занять больше времени, но обеспечит более высокую плотность.")
    
    def on_technology_changed(self, technology):
        """Обработчик изменения технологии"""
        self.technology = technology
        QMessageBox.information(self, "Технология изменена", 
                               f"Установлена технология: {technology}")
    
    def enable_controls(self, enable: bool):
        self.start_button.setEnabled(enable)
        self.pause_button.setEnabled(not enable)  # Пауза доступна только во время работы
        self.stop_button.setEnabled(not enable)
        self.profile_combo.setEnabled(enable)
        self.mode_combo.setEnabled(enable)
        self.tech_combo.setEnabled(enable)
        self.gap_spin.setEnabled(enable)
        self.time_spin.setEnabled(enable)
        self.reset_button.setEnabled(enable)
    
    def enable_export(self, enable: bool):
        self.dxf_button.setEnabled(enable)
        self.svg_button.setEnabled(enable)
        self.json_button.setEnabled(enable)
    
    def update_progress(self, progress: int, status: str, utilization: Optional[float] = None, energy: float = 0.0):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)
        if utilization is not None:
            self.utilization_label.setText(f"Использование: {utilization:.1f}%")
        self.energy_label.setText(f"Энергия: {energy:.2f}")


class MonitoringPanel(QWidget):
    """Панель мониторинга процесса оптимизации"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Вкладки для разных типов мониторинга
        self.tabs = QTabWidget()
        
        # Вкладка: Статистика процесса
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        # Таблица задач
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(4)
        self.tasks_table.setHorizontalHeaderLabels(
            ["ID задачи", "Статус", "Прогресс", "Сообщение"]
        )
        self.tasks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(QLabel("Текущая задача:"))
        stats_layout.addWidget(self.tasks_table)
        
        stats_tab.setLayout(stats_layout)
        self.tabs.addTab(stats_tab, "Статистика")
        
        # Вкладка: Лог событий
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))
        
        log_layout.addWidget(QLabel("Лог событий:"))
        log_layout.addWidget(self.log_text)
        
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "Лог")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def update_task_status(self, task_id: str, status: str, progress: int, message: str):
        """Обновление статуса задачи"""
        self.tasks_table.setRowCount(1)
        self.tasks_table.setItem(0, 0, QTableWidgetItem(task_id))
        self.tasks_table.setItem(0, 1, QTableWidgetItem(status))
        self.tasks_table.setItem(0, 2, QTableWidgetItem(f"{progress}%"))
        self.tasks_table.setItem(0, 3, QTableWidgetItem(message))
    
    def add_log_message(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def clear_log(self):
        """Очистка лога"""
        self.log_text.clear()


class MonitoringPanel(QWidget):
    """
    Панель мониторинга процесса оптимизации ИАГИ
    Включает статистику, графики сходимости и лог событий
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Инициализация элементов мониторинга"""
        layout = QVBoxLayout()
        
        # Вкладки для разных типов мониторинга
        self.tabs = QTabWidget()
        
        # Вкладка: Статистика процесса
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        # Таблица агентов/фигур
        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(6)
        self.agents_table.setHorizontalHeaderLabels(
            ["ID", "Имя", "Приоритет", "Положение", "Угол", "Статус"]
        )
        self.agents_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(QLabel("Состояние фигур:"))
        stats_layout.addWidget(self.agents_table)
        
        # Статистика энергии
        energy_layout = QHBoxLayout()
        
        self.total_energy_label = QLabel("Полная энергия: 0.0")
        energy_layout.addWidget(self.total_energy_label)
        
        self.kinetic_energy_label = QLabel("Кинетическая: 0.0")
        energy_layout.addWidget(self.kinetic_energy_label)
        
        self.potential_energy_label = QLabel("Потенциальная: 0.0")
        energy_layout.addWidget(self.potential_energy_label)
        
        stats_layout.addLayout(energy_layout)
        
        stats_tab.setLayout(stats_layout)
        self.tabs.addTab(stats_tab, "Статистика")
        
        # Вкладка: Графики сходимости
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout()
        
        # График изменения энергии
        self.energy_fig = Figure(figsize=(5, 3), dpi=100)
        self.energy_canvas = FigureCanvas(self.energy_fig)
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_ax.set_title('Изменение энергии системы')
        self.energy_ax.set_xlabel('Итерация')
        self.energy_ax.set_ylabel('Энергия')
        self.energy_ax.grid(True)
        
        graphs_layout.addWidget(self.energy_canvas)
        
        # График изменения коэффициента использования
        self.utilization_fig = Figure(figsize=(5, 3), dpi=100)
        self.utilization_canvas = FigureCanvas(self.utilization_fig)
        self.utilization_ax = self.utilization_fig.add_subplot(111)
        self.utilization_ax.set_title('Коэффициент использования материала')
        self.utilization_ax.set_xlabel('Итерация')
        self.utilization_ax.set_ylabel('η (%)')
        self.utilization_ax.grid(True)
        
        graphs_layout.addWidget(self.utilization_canvas)
        
        graphs_tab.setLayout(graphs_layout)
        self.tabs.addTab(graphs_tab, "Графики")
        
        # Вкладка: Лог событий
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))
        
        log_layout.addWidget(QLabel("Лог событий:"))
        log_layout.addWidget(self.log_text)
        
        # Кнопка очистки лога
        clear_button = QPushButton("Очистить лог")
        clear_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_button)
        
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "Лог")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def update_agents_table(self, shapes_data: List[Dict]):
        """Обновление таблицы фигур"""
        self.agents_table.setRowCount(len(shapes_data))
        
        for i, shape in enumerate(shapes_data):
            # ID
            self.agents_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # Имя
            name = shape.get('name', f'Part_{i+1}')
            self.agents_table.setItem(i, 1, QTableWidgetItem(name))
            
            # Приоритет
            priority = shape.get('priority', 2)
            self.agents_table.setItem(i, 2, QTableWidgetItem(str(priority)))
            
            # Положение
            centroid = shape.get('centroid', (0, 0))
            position = f"({centroid[0]:.1f}, {centroid[1]:.1f})"
            self.agents_table.setItem(i, 3, QTableWidgetItem(position))
            
            # Угол
            angle = shape.get('angle', 0)
            self.agents_table.setItem(i, 4, QTableWidgetItem(f"{angle:.1f}°"))
            
            # Статус
            status = "Размещена"
            self.agents_table.setItem(i, 5, QTableWidgetItem(status))
    
    def update_energy_graphs(self, iterations: List[int], energies: List[float], utilizations: List[float]):
        """Обновление графиков энергии и использования"""
        if len(iterations) < 2:
            return
            
        self.energy_ax.clear()
        self.energy_ax.plot(iterations, energies, 'b-', linewidth=2)
        self.energy_ax.set_title('Изменение энергии системы')
        self.energy_ax.set_xlabel('Итерация')
        self.energy_ax.set_ylabel('Энергия')
        self.energy_ax.grid(True)
        self.energy_canvas.draw()
        
        self.utilization_ax.clear()
        self.utilization_ax.plot(iterations, utilizations, 'g-', linewidth=2)
        self.utilization_ax.set_title('Коэффициент использования материала')
        self.utilization_ax.set_xlabel('Итерация')
        self.utilization_ax.set_ylabel('η (%)')
        self.utilization_ax.grid(True)
        self.utilization_canvas.draw()
    
    def update_energy_stats(self, total_energy: float, kinetic_energy: float, potential_energy: float):
        """Обновление статистики энергии"""
        self.total_energy_label.setText(f"Полная энергия: {total_energy:.2f}")
        self.kinetic_energy_label.setText(f"Кинетическая: {kinetic_energy:.2f}")
        self.potential_energy_label.setText(f"Потенциальная: {potential_energy:.2f}")
    
    def add_log_message(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def clear_log(self):
        """Очистка лога"""
        self.log_text.clear()


class LoadFileDialog(QDialog):
    """
    Диалог для загрузки данных из файлов
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Загрузка данных")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Выбор файла
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Файл данных:"))
        
        self.file_edit = QLineEdit()
        file_layout.addWidget(self.file_edit)
        
        self.browse_button = QPushButton("Обзор...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        
        layout.addLayout(file_layout)
        
        # Информация о файле
        info_group = QGroupBox("Информация о файле")
        info_layout = QVBoxLayout()
        
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setFixedHeight(100)
        info_layout.addWidget(self.file_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Настройки импорта
        import_group = QGroupBox("Настройки импорта")
        import_layout = QFormLayout()
        
        self.sheet_width = QDoubleSpinBox()
        self.sheet_width.setRange(100, 10000)
        self.sheet_width.setValue(2000)
        import_layout.addRow("Ширина листа (мм):", self.sheet_width)
        
        self.sheet_height = QDoubleSpinBox()
        self.sheet_height.setRange(100, 10000)
        self.sheet_height.setValue(1000)
        import_layout.addRow("Высота листа (мм):", self.sheet_height)
        
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.01, 1.0)
        self.tolerance_spin.setValue(0.1)
        self.tolerance_spin.setSingleStep(0.01)
        import_layout.addRow("Точность (мм):", self.tolerance_spin)
        
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def browse_file(self):
        """Выбор файла"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", 
            "DXF Files (*.dxf);;STEP Files (*.step *.stp);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            self.file_edit.setText(filename)
            self.update_file_info(filename)
    
    def update_file_info(self, filename: str):
        """Обновление информации о файле"""
        import os
        
        if not os.path.exists(filename):
            self.file_info.setText("Файл не найден")
            return
        
        # Получение информации о файле
        file_size = os.path.getsize(filename) / 1024  # KB
        file_ext = os.path.splitext(filename)[1].lower()
        
        info = f"Имя файла: {os.path.basename(filename)}\n"
        info += f"Размер: {file_size:.1f} KB\n"
        info += f"Тип: {file_ext.upper()[1:]}\n"
        
        if file_ext == '.dxf' or file_ext == '.step':
            info += "Формат: CAD-файл с геометрией деталей"
        elif file_ext == '.json':
            info += "Формат: Конфигурационный файл"
        
        self.file_info.setText(info)


class SplashScreen(QDialog):
    """
    Экран-заставка с описанием этапов запуска
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(500, 400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Основной контейнер с фоном
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border-radius: 15px;
                border: 2px solid #5A9FD5;
            }
        """)
        container_layout = QVBoxLayout()
        
        # Заголовок
        title_label = QLabel("ИАГИ Клиент")
        title_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title_label)
        
        # Список этапов
        self.steps_list = QListWidget()
        self.steps_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                color: #E0E0E0;
                font-size: 14px;
                border: none;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #555555;
            }
            QListWidget::item:selected {
                background-color: #5A9FD5;
            }
        """)
        
        stages = [
            "⏳ Инициализация приложения...",
            "📂 Загрузка настроек...",
            "🔌 Подключение к серверу...",
            "🎨 Применение темы...",
            "✅ Готово к работе"
        ]
        
        for stage in stages:
            item = QListWidgetItem(stage)
            self.steps_list.addItem(item)
        
        container_layout.addWidget(self.steps_list)
        
        # Индикатор прогресса
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3D3D3D;
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #5A9FD5;
                border-radius: 4px;
            }
        """)
        container_layout.addWidget(self.progress_bar)
        
        # Статус
        self.status_label = QLabel("Запуск...")
        self.status_label.setStyleSheet("""
            color: #AAAAAA;
            font-size: 12px;
            padding: 10px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.status_label)
        
        container.setLayout(container_layout)
        layout.addWidget(container)
        
        self.setLayout(layout)
    
    def update_step(self, step_index: int, progress: int, status_text: str):
        """Обновление текущего шага"""
        # Выделение текущего шага
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            if i == step_index:
                self.steps_list.setCurrentItem(item)
                item.setForeground(QColor("#5A9FD5"))
            else:
                item.setForeground(QColor("#E0E0E0"))
        
        self.progress_bar.setValue(progress)
        self.status_label.setText(status_text)
        QApplication.processEvents()


class MainWindow(QMainWindow):
    """
    Главное окно клиентского приложения ИАГИ
    Объединяет функционал локального интерфейса с удалённым доступом к серверу
    """
    
    def __init__(self, server_url: str = "http://localhost:8000", theme: str = "system"):
        super().__init__()
        self.setWindowTitle("ИАГИ Клиент - Раскрой плоских деталей")
        self.setGeometry(100, 100, 1400, 900)
        
        # Настройки
        self.settings = QSettings("IAGI", "Client")
        self.server_url = server_url or self.settings.value("server_url", "http://localhost:8000")
        self.current_theme = theme or self.settings.value("theme", "system")
        
        # Импорт APIClient здесь, чтобы избежать циклических зависимостей
        from client.main import APIClient, OptimizationThread
        self.api_client = APIClient(self.server_url)
        self.OptimizationThread = OptimizationThread
        
        # Текущие данные
        self.current_file = None
        self.current_task_id = None
        self.optimization_thread: Optional[Any] = None
        self.current_result_data = None
        
        self.init_ui()
        self.apply_theme(self.current_theme)
        self.check_server_connection()
    
    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель: управление
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Выбор файла
        file_group = QGroupBox("Входные данные")
        file_layout = QVBoxLayout()
        
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("font-style: italic;")
        file_layout.addWidget(self.file_label)
        
        file_button_layout = QHBoxLayout()
        self.browse_button = QPushButton("📁 Выбрать файл...")
        self.browse_button.clicked.connect(self.browse_file)
        file_button_layout.addWidget(self.browse_button)
        
        self.shapes_info_label = QLabel("")
        file_button_layout.addWidget(self.shapes_info_label)
        
        file_layout.addLayout(file_button_layout)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Панель управления
        self.control_panel = ControlPanel()
        self.control_panel.start_button.clicked.connect(self.start_optimization)
        self.control_panel.stop_button.clicked.connect(self.stop_optimization)
        self.control_panel.dxf_button.clicked.connect(lambda: self.download_result('dxf'))
        self.control_panel.svg_button.clicked.connect(lambda: self.download_result('svg'))
        self.control_panel.json_button.clicked.connect(lambda: self.download_result('json'))
        left_layout.addWidget(self.control_panel)
        
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        
        # Центральная панель: визуализация
        center_panel = QGroupBox("Визуализация")
        center_layout = QVBoxLayout()
        
        self.visualization = RealTimeVisualization(
            parent=self,
            sheet_size=(2000, 1000),
            width=12, height=8, dpi=100
        )
        center_layout.addWidget(self.visualization)
        
        # Кнопки управления визуализацией
        vis_controls = QHBoxLayout()
        
        self.refresh_vis_button = QPushButton("🔄 Обновить")
        self.refresh_vis_button.clicked.connect(self.refresh_visualization)
        vis_controls.addWidget(self.refresh_vis_button)
        
        self.export_png_button = QPushButton("💾 PNG")
        self.export_png_button.clicked.connect(self.export_visualization_png)
        vis_controls.addWidget(self.export_png_button)
        
        self.export_svg_button = QPushButton("📄 SVG")
        self.export_svg_button.clicked.connect(self.export_visualization_svg)
        vis_controls.addWidget(self.export_svg_button)
        
        vis_controls.addStretch()
        center_layout.addLayout(vis_controls)
        
        center_panel.setLayout(center_layout)
        
        # Правая панель: мониторинг
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.monitoring_panel = MonitoringPanel()
        right_layout.addWidget(self.monitoring_panel)
        
        # Информация о задаче
        info_group = QGroupBox("Информация о задаче")
        info_layout = QVBoxLayout()
        
        self.task_info = QTextEdit()
        self.task_info.setReadOnly(True)
        self.task_info.setMaximumHeight(150)
        info_layout.addWidget(self.task_info)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        
        # Добавление панелей в основной макет
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 700, 350])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Меню
        self.create_menu()
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
    
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('Файл')
        
        connect_action = file_menu.addAction('Подключение к серверу...')
        connect_action.triggered.connect(self.show_connection_dialog)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Выход')
        exit_action.triggered.connect(self.close)
        
        # Меню настроек
        settings_menu = menubar.addMenu('Настройки')
        
        theme_menu = settings_menu.addMenu('Цветовая тема')
        
        self.theme_system_action = theme_menu.addAction('Системная')
        self.theme_system_action.setCheckable(True)
        self.theme_system_action.setChecked(self.current_theme == 'system')
        self.theme_system_action.triggered.connect(lambda: self.apply_theme('system'))
        
        self.theme_light_action = theme_menu.addAction('Светлая')
        self.theme_light_action.setCheckable(True)
        self.theme_light_action.setChecked(self.current_theme == 'light')
        self.theme_light_action.triggered.connect(lambda: self.apply_theme('light'))
        
        self.theme_dark_action = theme_menu.addAction('Тёмная')
        self.theme_dark_action.setCheckable(True)
        self.theme_dark_action.setChecked(self.current_theme == 'dark')
        self.theme_dark_action.triggered.connect(lambda: self.apply_theme('dark'))
        
        help_menu = menubar.addMenu('Помощь')
        
        about_action = help_menu.addAction('О программе')
        about_action.triggered.connect(self.show_about)
    
    def apply_theme(self, theme: str):
        """Применяет цветовую тему"""
        self.current_theme = theme
        self.settings.setValue("theme", theme)
        
        # Импорт функций темы
        from client.themes import get_palette, generate_stylesheet
        
        palette = get_palette(theme)
        stylesheet = generate_stylesheet(palette)
        self.setStyleSheet(stylesheet)
        
        # Обновляем чекбоксы в меню
        if hasattr(self, 'theme_system_action'):
            self.theme_system_action.setChecked(theme == 'system')
            self.theme_light_action.setChecked(theme == 'light')
            self.theme_dark_action.setChecked(theme == 'dark')
        
        self.monitoring_panel.add_log_message(f"Применена тема: {theme}")
    
    def show_connection_dialog(self):
        dialog = ServerConnectionDialog(self, self.server_url, self.current_theme)
        if dialog.exec_() == QDialog.Accepted:
            self.server_url = dialog.get_server_url()
            self.settings.setValue("server_url", self.server_url)
            self.api_client = self.api_client.__class__(self.server_url)
            
            # Применяем новую тему если она изменилась
            new_theme = dialog.get_theme()
            if new_theme != self.current_theme:
                self.apply_theme(new_theme)
            
            self.check_server_connection()
    
    def check_server_connection(self):
        if self.api_client.check_connection():
            self.connection_label_text = f"Сервер: {self.server_url} ✓"
            self.statusBar().showMessage(f"Подключено к {self.server_url}")
            self.monitoring_panel.add_log_message("Подключение к серверу успешно")
        else:
            self.connection_label_text = "Сервер: Не подключено"
            self.statusBar().showMessage("Сервер недоступен")
            self.monitoring_panel.add_log_message("Не удалось подключиться к серверу")
    
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с фигурами",
            "",
            "DXF Files (*.dxf);;STEP Files (*.step *.stp);;All Files (*)"
        )
        
        if filename:
            self.current_file = filename
            self.file_label.setText(Path(filename).name)
            self.monitoring_panel.add_log_message(f"Выбран файл: {filename}")
            
            # Получение информации о фигурах
            try:
                shapes_info = self.api_client.get_shapes_info(filename)
                num_shapes = shapes_info.get('num_shapes', 0)
                self.shapes_info_label.setText(f"{num_shapes} фигур")
                self.control_panel.start_button.setEnabled(True)
                self.monitoring_panel.add_log_message(f"Загружено {num_shapes} фигур")
            except Exception as e:
                self.shapes_info_label.setText("Ошибка")
                self.monitoring_panel.add_log_message(f"Ошибка получения информации: {e}")
                self.control_panel.start_button.setEnabled(False)
    
    def start_optimization(self):
        if not self.current_file:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл для оптимизации")
            return
        
        self.control_panel.enable_controls(False)
        self.control_panel.enable_export(False)
        
        try:
            task_id = self.api_client.start_optimization(
                file_path=self.current_file,
                profile=self.control_panel.profile_combo.currentText(),
                algorithm=self.control_panel.mode_combo.currentText(),
                technology=self.control_panel.tech_combo.currentText(),
                min_gap=self.control_panel.gap_spin.value(),
                time_limit=self.control_panel.time_spin.value(),
                sheet_width=self.control_panel.sheet_width_spin.value(),
                sheet_height=self.control_panel.sheet_height_spin.value(),
                use_original_positions=True
            )
            
            self.current_task_id = task_id
            self.monitoring_panel.add_log_message(f"Задача запущена: {task_id}")
            self.task_info.setText(f"ID задачи: {task_id}\nСтатус: В ожидании...")
            
            # Запуск мониторинга
            self.optimization_thread = self.OptimizationThread(self.api_client, task_id)
            self.optimization_thread.status_update.connect(self.on_status_update)
            self.optimization_thread.finished_signal.connect(self.on_optimization_finished)
            self.optimization_thread.error_signal.connect(self.on_optimization_error)
            self.optimization_thread.start()
            
            self.control_panel.stop_button.setEnabled(True)
            self.statusBar().showMessage("Оптимизация запущена...")
            
        except Exception as e:
            self.monitoring_panel.add_log_message(f"Ошибка запуска: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить оптимизацию: {e}")
            self.control_panel.enable_controls(True)
    
    def on_status_update(self, status: dict):
        utilization = status.get('utilization')
        self.control_panel.update_progress(
            status.get('progress', 0),
            status.get('message', ''),
            utilization
        )
        
        self.task_info.setText(
            f"ID задачи: {self.current_task_id}\n"
            f"Статус: {status.get('status', 'unknown')}\n"
            f"Прогресс: {status.get('progress', 0)}%\n"
            f"Сообщение: {status.get('message', '')}"
        )
        
        self.monitoring_panel.update_task_status(
            self.current_task_id,
            status.get('status', 'unknown'),
            status.get('progress', 0),
            status.get('message', '')
        )
    
    def on_optimization_finished(self, status: dict):
        self.monitoring_panel.add_log_message(f"Оптимизация завершена: {status.get('message', '')}")
        self.control_panel.enable_export(True)
        self.control_panel.stop_button.setEnabled(False)
        
        utilization = status.get('utilization') if status else 0
        if utilization is None:
            utilization = 0
        
        self.statusBar().showMessage(f"Оптимизация завершена. Использование: {utilization:.1f}%")
        
        # Попытка получить данные для визуализации
        try:
            result_info = self.api_client.get_task_status(self.current_task_id)
            if 'result' in result_info:
                self.current_result_data = result_info['result']
                self.refresh_visualization()
        except Exception:
            pass
    
    def on_optimization_error(self, error: str):
        self.monitoring_panel.add_log_message(f"Ошибка: {error}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка оптимизации: {error}")
        self.control_panel.enable_controls(True)
    
    def stop_optimization(self):
        if self.optimization_thread:
            self.optimization_thread.stop()
            self.monitoring_panel.add_log_message("Оптимизация остановлена пользователем")
            self.control_panel.enable_controls(True)
    
    def download_result(self, format_type: str):
        if not self.current_task_id:
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "Выберите директорию для сохранения"
        )
        
        if not output_dir:
            return
        
        try:
            output_path = Path(output_dir) / f"iagi_result.{format_type}"
            self.api_client.download_result(self.current_task_id, format_type, str(output_path))
            self.monitoring_panel.add_log_message(f"Результат сохранен: {output_path}")
            QMessageBox.information(self, "Экспорт завершен", f"Файл сохранен:\n{output_path}")
        except Exception as e:
            self.monitoring_panel.add_log_message(f"Ошибка экспорта: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результат: {e}")
    
    def refresh_visualization(self):
        """Обновление визуализации с данными от сервера"""
        if not self.current_task_id:
            return
        
        try:
            status = self.api_client.get_task_status(self.current_task_id)
            utilization = status.get('utilization', 0.0) or 0.0
            
            # Если есть данные о размещённых фигурах
            if 'result' in status and isinstance(status['result'], list):
                self.visualization.update_visualization(status['result'], utilization)
            else:
                # Пустая визуализация с размерами листа
                self.visualization.update_visualization([], utilization)
                
        except Exception as e:
            self.monitoring_panel.add_log_message(f"Ошибка обновления визуализации: {e}")
    
    def export_visualization_png(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить визуализацию", "", "PNG Files (*.png)"
        )
        if filename:
            self.visualization.export_to_png(filename)
            self.monitoring_panel.add_log_message(f"Визуализация экспортирована: {filename}")
    
    def export_visualization_svg(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить визуализацию", "", "SVG Files (*.svg)"
        )
        if filename:
            self.visualization.export_to_svg(filename)
            self.monitoring_panel.add_log_message(f"Визуализация экспортирована: {filename}")
    
    def show_about(self):
        QMessageBox.about(
            self, "О программе",
            "ИАГИ Клиент\n\n"
            "Клиентское приложение для системы оптимизации раскроя\n"
            "плоских деталей с использованием гравитационной имитации.\n\n"
            "Версия 2.0\n"
            "Подключается к серверу ИАГИ через HTTP API\n"
            "Включает визуализацию результатов и мониторинг процесса"
        )
