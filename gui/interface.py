import sys
import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon, Rectangle, Circle, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from core.geometry import PolygonShape
from core.constraints import ConstraintManager
from core.optimizer import PackingOptimizer
from my_io.exporter import export_results
from config.settings import get_config, load_profile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QTabWidget, QFileDialog, QProgressBar, QMessageBox,
                             QTextEdit, QSplitter, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QDialog, QFormLayout, QDialogButtonBox, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPalette, QLinearGradient, QBrush, QPainter, QPen, QPolygonF, QFont

class RealTimeVisualization(FigureCanvas):
    """
    Виджет для визуализации процесса раскроя в реальном времени
    Согласно разделу 3.1.5, визуализация является критически важной для интерпретируемости
    и понимания поведения агентов в динамической системе ИАГИ.
    """
    
    def __init__(self, parent=None, sheet_size=(2000, 1000), width=10, height=8, dpi=100):
        """
        Инициализация визуализации
        
        :param parent: Родительский виджет
        :param sheet_size: Размеры листа (ширина, высота) в мм
        :param width, height: Размеры холста в дюймах
        :param dpi: Разрешение холста
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.sheet_size = sheet_size
        self.current_positions = {}
        self.agents = []
        self.defect_zones = []
        self.constraint_manager = None
        self.show_forces = False
        self.show_velocities = False
        self.show_sdf = False
        self.sdf_fields = {}
        self.color_map = {}
        
        self.setup_plot()
    
    def setup_plot(self):
        """Настройка начального состояния графика"""
        self.axes.set_xlim(0, self.sheet_size[0])
        self.axes.set_ylim(0, self.sheet_size[1])
        self.axes.set_aspect('equal')
        self.axes.set_title('Процесс раскроя ИАГИ', fontsize=14)
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
    
    def update_visualization(self, agents: List[Any], defect_zones: List[PolygonShape] = None,
                           forces: Dict[int, np.ndarray] = None, 
                           constraint_manager: ConstraintManager = None):
        """
        Обновление визуализации в реальном времени
        
        :param agents: Список агентов с текущими позициями и состояниями
        :param defect_zones: Список дефектных зон
        :param forces: Словарь сил, действующих на агентов
        :param constraint_manager: Менеджер технологических ограничений
        """
        self.axes.clear()
        self.setup_plot()
        
        # Сохранение ссылок для последующего использования
        self.agents = agents
        self.defect_zones = defect_zones if defect_zones else []
        self.constraint_manager = constraint_manager
        self.forces = forces if forces else {}
        
        # Отрисовка дефектных зон
        # self._draw_defect_zones()
        
        # Отрисовка фигур по приоритетам (для корректного наложения)
        max_priority = max(agent.priority for agent in agents) if agents else 1
        
        for priority in range(max_priority, 0, -1):
            for agent in agents:
                if agent.priority == priority:
                    self._draw_agent(agent, priority)
        
        # Отрисовка сил и скоростей, если включено
        if self.show_forces:
            self._draw_forces()
        
        if self.show_velocities:
            self._draw_velocities()
        
        # Обновление легенды
        self._update_legend()
        
        # Перерисовка
        self.draw()
    
    def _draw_agent(self, agent, priority):
        """Отрисовка отдельного агента"""
        # Получение трансформированной фигуры
        transformed_shape = agent.get_transformed_shape()
        
        # Цвета на основе приоритета и состояния
        if agent.is_frozen:
            fill_alpha = 0.6
            edge_color = 'darkgreen'
        else:
            fill_alpha = 0.8
            edge_color = self._get_priority_color(priority)
        
        # Основной контур фигуры
        polygon = MplPolygon(transformed_shape.outer_contour, closed=True,
                           fill=True, facecolor=self._get_priority_color(priority, fill_alpha),
                           edgecolor=edge_color, linewidth=1.5, zorder=10)
        self.axes.add_patch(polygon)
        
        # Внутренние контуры (отверстия)
        for inner_contour in transformed_shape.inner_contours:
            hole = MplPolygon(inner_contour, closed=True,
                            fill=True, facecolor='white',
                            edgecolor=edge_color, linewidth=1.0, linestyle='--',
                            zorder=11)
            self.axes.add_patch(hole)
        
        # Центр масс и направление
        centroid = transformed_shape.centroid
        self.axes.plot(centroid[0], centroid[1], 'ko', markersize=4, zorder=12)
        
        # Направление (ось X)
        direction = np.array([np.cos(np.radians(agent.angle)), np.sin(np.radians(agent.angle))])
        end_point = centroid + direction * min(20, transformed_shape.get_bounding_box()[2] - centroid[0])
        self.axes.plot([centroid[0], end_point[0]], [centroid[1], end_point[1]], 
                      color='red', linewidth=2, zorder=12)
        
        # Подпись с именем и приоритетом
        name = getattr(agent.shape, 'name', f'Part_{agent.priority}')
        self.axes.text(centroid[0], centroid[1], f'{name}\nP={priority}', 
                      ha='center', va='center', fontsize=8,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                      zorder=13)
    
    def _draw_defect_zones(self):
        """Отрисовка дефектных зон"""
        for i, defect in enumerate(self.defect_zones):
            defect_polygon = MplPolygon(defect.outer_contour, closed=True,
                                      fill=True, facecolor='red', alpha=0.3,
                                      edgecolor='darkred', linewidth=1.5,
                                      linestyle='--', zorder=5)
            self.axes.add_patch(defect_polygon)
            
            # Текстовая метка для дефекта
            centroid = defect.centroid
            self.axes.text(centroid[0], centroid[1], f"ДЕФЕКТ {i+1}", 
                          ha='center', va='center', fontsize=8, color='darkred',
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                          zorder=6)
    
    def _draw_forces(self):
        """Отрисовка векторов сил"""
        scale = 0.5  # Масштаб для отображения сил
        
        for agent in self.agents:
            agent_id = id(agent)
            if agent_id in self.forces and not agent.is_frozen:
                force = self.forces[agent_id]
                position = agent.get_transformed_shape().centroid
                
                # Вектор силы
                end_point = position + force * scale
                self.axes.annotate('', xy=end_point, xytext=position,
                                 arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                                 zorder=15)
    
    def _draw_velocities(self):
        """Отрисовка векторов скоростей"""
        scale = 2.0  # Масштаб для отображения скоростей
        
        for agent in self.agents:
            if not agent.is_frozen:
                velocity = agent.velocity
                position = agent.get_transformed_shape().centroid
                
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
        self.update_visualization(self.agents, self.defect_zones, self.forces, self.constraint_manager)
    
    def toggle_velocities(self, show=True):
        """Переключение отображения скоростей"""
        self.show_velocities = show
        self.update_visualization(self.agents, self.defect_zones, self.forces, self.constraint_manager)
    
    def toggle_sdf(self, show=True):
        """Переключение отображения полей расстояний"""
        self.show_sdf = show
        self.update_visualization(self.agents, self.defect_zones, self.forces, self.constraint_manager)
    
    def export_to_svg(self, filename):
        """Экспорт текущей визуализации в SVG"""
        self.fig.savefig(filename, format='svg', bbox_inches='tight')
        print(f"Визуализация экспортирована в {filename}")
    
    def export_to_png(self, filename):
        """Экспорт текущей визуализации в PNG"""
        self.fig.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        print(f"Визуализация экспортирована в {filename}")

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
        self.config = get_config()
        
        # Текущие настройки
        self.current_profile = 'medium_precision'
        self.placement_mode = 'sequential'
        self.technology = 'laser'
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация элементов управления"""
        layout = QVBoxLayout()
        
        # Группа выбора профиля конфигурации
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
        
        # Группа режима размещения
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
        
        # Группа технологических параметров
        tech_group = QGroupBox("Технологические параметры")
        tech_layout = QVBoxLayout()
        
        # Технология резки
        tech_layout.addWidget(QLabel("Технология резки:"))
        self.tech_combo = QComboBox()
        self.tech_combo.addItems(['laser', 'plasma', 'waterjet', 'default'])
        self.tech_combo.setCurrentText(self.technology)
        self.tech_combo.currentTextChanged.connect(self.on_technology_changed)
        tech_layout.addWidget(self.tech_combo)
        
        # Минимальный зазор
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Минимальный зазор (мм):"))
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.1, 10.0)
        self.gap_spin.setValue(self.config.geometry['collision_detection']['min_gap'])
        self.gap_spin.setSingleStep(0.1)
        gap_layout.addWidget(self.gap_spin)
        tech_layout.addLayout(gap_layout)
        
        # Максимальное время
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Макс. время (сек):"))
        self.time_spin = QSpinBox()
        self.time_spin.setRange(10, 3600)
        self.time_spin.setValue(self.config.system['max_execution_time'])
        time_layout.addWidget(self.time_spin)
        tech_layout.addLayout(time_layout)
        
        tech_group.setLayout(tech_layout)
        layout.addWidget(tech_group)
        
        # Группа управления процессом
        control_group = QGroupBox("Управление процессом")
        control_layout = QVBoxLayout()
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Запустить")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.start_optimization.emit)
        
        self.pause_button = QPushButton("⏸ Пауза")
        self.pause_button.setStyleSheet("background-color: #FFC107; color: black;")
        self.pause_button.clicked.connect(self.pause_optimization.emit)
        
        self.stop_button = QPushButton("⏹ Остановить")
        self.stop_button.setStyleSheet("background-color: #F44336; color: white;")
        self.stop_button.clicked.connect(self.stop_optimization.emit)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        
        control_layout.addLayout(button_layout)
        
        # Индикатор прогресса
        self.progress_label = QLabel("Готов к запуску")
        control_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)
        
        # Статистика
        stats_layout = QHBoxLayout()
        
        self.utilization_label = QLabel("Использование: 0.0%")
        stats_layout.addWidget(self.utilization_label)
        
        self.energy_label = QLabel("Энергия: 0.0")
        stats_layout.addWidget(self.energy_label)
        
        control_layout.addLayout(stats_layout)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Группа экспорта
        export_group = QGroupBox("Экспорт результатов")
        export_layout = QVBoxLayout()
        
        export_layout.addWidget(QLabel("Форматы экспорта:"))
        
        export_button_layout = QHBoxLayout()
        
        self.dxf_button = QPushButton("Экспорт DXF")
        self.dxf_button.clicked.connect(lambda: self.export_results.emit('dxf'))
        export_button_layout.addWidget(self.dxf_button)
        
        self.svg_button = QPushButton("Экспорт SVG")
        self.svg_button.clicked.connect(lambda: self.export_results.emit('svg'))
        export_button_layout.addWidget(self.svg_button)
        
        self.json_button = QPushButton("Экспорт JSON")
        self.json_button.clicked.connect(lambda: self.export_results.emit('json'))
        export_button_layout.addWidget(self.json_button)
        
        export_layout.addLayout(export_button_layout)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Кнопка сброса
        self.reset_button = QPushButton("🔄 Сбросить симуляцию")
        self.reset_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.reset_button.clicked.connect(self.reset_simulation.emit)
        layout.addWidget(self.reset_button)
        
        self.setLayout(layout)
    
    def on_profile_changed(self, profile_name):
        """Обработчик изменения профиля"""
        self.current_profile = profile_name
        load_profile(profile_name)
        self.config = get_config()
        
        # Обновление зазора в соответствии с профилем
        min_gap = self.config.technological_constraints['min_gaps'].get(
            self.config.technological_constraints['cutting_technology'], 
            self.config.geometry['collision_detection']['min_gap']
        )
        self.gap_spin.setValue(min_gap)
        
        QMessageBox.information(self, "Профиль изменен", 
                               f"Применен профиль: {profile_name}\n"
                               f"Точность: {self.config.geometry['polygonization']['tolerance']} мм\n"
                               f"Минимальный зазор: {min_gap} мм")
    
    def on_mode_changed(self, mode):
        """Обработчик изменения режима"""
        self.placement_mode = mode
        self.config.set_placement_mode(mode)
        
        if mode == 'hybrid':
            QMessageBox.information(self, "Гибридный режим", 
                                   "Включены гибридные алгоритмы (ГА-ИАГИ, PSO-ИАГИ).\n"
                                   "Оптимизация может занять больше времени, но обеспечит более высокую плотность.")
    
    def on_technology_changed(self, technology):
        """Обработчик изменения технологии"""
        self.technology = technology
        self.config.set_technology(technology)
        
        # Обновление зазора
        min_gap = self.config.geometry['collision_detection']['min_gap']
        self.gap_spin.setValue(min_gap)
        
        QMessageBox.information(self, "Технология изменена", 
                               f"Установлена технология: {technology}\n"
                               f"Минимальный зазор: {min_gap} мм")
    
    def update_progress(self, progress: int, status: str, utilization: float = 0.0, energy: float = 0.0):
        """Обновление прогресса и статуса"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)
        self.utilization_label.setText(f"Использование: {utilization:.1f}%")
        self.energy_label.setText(f"Энергия: {energy:.2f}")
    
    def enable_controls(self, enable: bool):
        """Включение/выключение элементов управления"""
        self.start_button.setEnabled(enable)
        self.pause_button.setEnabled(not enable)  # Пауза доступна только во время работы
        self.stop_button.setEnabled(not enable)
        self.profile_combo.setEnabled(enable)
        self.mode_combo.setEnabled(enable)
        self.tech_combo.setEnabled(enable)
        self.gap_spin.setEnabled(enable)
        self.time_spin.setEnabled(enable)
        self.reset_button.setEnabled(enable)

class MonitoringPanel(QWidget):
    """
    Панель мониторинга процесса оптимизации ИАГИ
    Согласно разделу 3.5.7, визуализация состояния системы и анализ сходимости
    являются критически важными для понимания и настройки динамической модели.
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
        
        # Таблица агентов
        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(6)
        self.agents_table.setHorizontalHeaderLabels(
            ["ID", "Имя", "Приоритет", "Положение", "Скорость", "Статус"]
        )
        self.agents_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_layout.addWidget(QLabel("Состояние агентов:"))
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
        
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "Лог")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def update_agents_table(self, agents):
        """Обновление таблицы агентов"""
        self.agents_table.setRowCount(len(agents))
        
        for i, agent in enumerate(agents):
            # ID
            self.agents_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            
            # Имя
            name = getattr(agent.shape, 'name', f'Part_{i+1}')
            self.agents_table.setItem(i, 1, QTableWidgetItem(name))
            
            # Приоритет
            self.agents_table.setItem(i, 2, QTableWidgetItem(str(agent.priority)))
            
            # Положение
            position = f"({agent.position[0]:.1f}, {agent.position[1]:.1f})"
            self.agents_table.setItem(i, 3, QTableWidgetItem(position))
            
            # Скорость
            speed = np.linalg.norm(agent.velocity)
            self.agents_table.setItem(i, 4, QTableWidgetItem(f"{speed:.2f} мм/с"))
            
            # Статус
            status = "Заморожен" if agent.is_frozen else "Активен"
            self.agents_table.setItem(i, 5, QTableWidgetItem(status))
    
    def update_energy_graphs(self, iterations, total_energies, utilizations):
        """Обновление графиков энергии и использования"""
        self.energy_ax.clear()
        self.energy_ax.plot(iterations, total_energies, 'b-', linewidth=2)
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
    
    def update_energy_stats(self, total_energy, kinetic_energy, potential_energy):
        """Обновление статистики энергии"""
        self.total_energy_label.setText(f"Полная энергия: {total_energy:.2f}")
        self.kinetic_energy_label.setText(f"Кинетическая: {kinetic_energy:.2f}")
        self.potential_energy_label.setText(f"Потенциальная: {potential_energy:.2f}")
    
    def add_log_message(self, message):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("%H:%M:%S")
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
    
    def update_file_info(self, filename):
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

class OptimizationThread(QThread):
    """
    Поток для выполнения оптимизации в фоновом режиме
    """
    update_signal = pyqtSignal(list, list, dict, float, float, int)
    finished_signal = pyqtSignal(list, float)
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, optimizer, shapes, constraint_manager):
        super().__init__()
        self.optimizer:PackingOptimizer = optimizer
        self.shapes = shapes
        self.constraint_manager = constraint_manager
        self.running = True
        self.paused = False
        self.iteration = 0
        
    def run(self):
        """Основной метод выполнения оптимизации"""
        try:
            self.log_signal.emit("Запуск оптимизации ИАГИ...")
            
            # Подготовка начальных агентов
            agents = self.optimizer.prepare_agents(
                self.shapes,
                priorities=[getattr(shape, 'priority', 1) for shape in self.shapes],
                orientation_constraints=[getattr(shape, 'allowed_angles', None) for shape in self.shapes]
            )
            
            # Установка дефектных зон
            for defect in self.constraint_manager.defect_zones:
                self.optimizer.add_defect_zone(defect['polygon'])
            
            # Начальная визуализация
            self.update_visualization(agents, 0.0, 0.0)
            
            # Сохранение начального состояния для графиков
            iterations = [0]
            energies = [100.0]  # Начальная энергия (относительная)
            utilizations = [0.0]
            
            # Основной цикл оптимизации
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < self.optimizer.time_limit:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Один шаг оптимизации
                # Здесь должен быть код вызова одного шага ИАГИ
                # В реальной реализации это будет интеграция с core/optimizer.py
                
                # Имитация прогресса для демонстрации
                self.iteration += 1
                progress = min(100, int((self.iteration / 100) * 100))
                utilization = min(95.0, 50.0 + self.iteration * 0.45)
                energy = max(0.1, 100.0 - self.iteration * 0.9)
                
                # Обновление визуализации
                if self.iteration % 5 == 0:
                    self.update_visualization(agents, utilization, energy)
                
                # Обновление графиков
                if self.iteration % 10 == 0:
                    iterations.append(self.iteration)
                    energies.append(energy)
                    utilizations.append(utilization)
                    self.update_graphs(iterations, energies, utilizations)
                
                # Добавление сообщений в лог
                if self.iteration % 20 == 0:
                    self.log_signal.emit(f"Итерация {self.iteration}: использование={utilization:.1f}%")
                
                # Проверка завершения
                if progress >= 100 or utilization > 95.0:
                    break
                
                # Задержка для визуализации
                time.sleep(0.05)
            
            # Финальное обновление
            total_time = time.time() - start_time
            self.log_signal.emit(f"Оптимизация завершена за {total_time:.2f} секунд")
            
            # Финальная визуализация и результаты
            final_utilization = min(98.0, 70.0 + self.iteration * 0.28)
            self.finished_signal.emit(agents, final_utilization)
            
        except Exception as e:
            self.error_signal.emit(f"Ошибка оптимизации: {str(e)}")
    
    def update_visualization(self, agents, utilization, energy):
        """Обновление визуализации"""
        forces = {}  # Здесь должны быть реальные силы
        self.update_signal.emit(
            agents, 
            self.constraint_manager.defect_zones,
            forces,
            utilization,
            energy,
            self.iteration
        )
    
    def update_graphs(self, iterations, energies, utilizations):
        """Обновление графиков сходимости"""
        # Этот метод будет вызываться из основного потока
        pass
    
    def pause(self):
        """Пауза оптимизации"""
        self.paused = True
    
    def resume(self):
        """Возобновление оптимизации"""
        self.paused = False
    
    def stop(self):
        """Остановка оптимизации"""
        self.running = False

class MainWindow(QMainWindow):
    """
    Главное окно приложения ИАГИ
    Согласно разделу 3.6.5, интерфейс должен обеспечивать интеграцию с промышленными
    CAD/CAM-системами и предоставлять инструменты для анализа и верификации результатов.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ИАГИ - Интеллектуальный агент гравитационной имитации для раскроя")
        self.setGeometry(100, 100, 1600, 900)
        
        # Загрузка конфигурации
        self.config = get_config()
        
        # Инициализация компонентов
        self.constraint_manager = ConstraintManager(self.config.get_sheet_size_from_input(""))
        self.optimizer:PackingOptimizer = None
        
        # Текущие данные
        self.current_shapes = []
        self.current_agents = []
        self.current_defect_zones = []
        
        # Инициализация UI
        self.init_ui()
        
        # Таймер для обновления интерфейса
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # Обновление 10 раз в секунду
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Центральный виджет
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель: визуализация
        left_panel = QGroupBox("Визуализация процесса раскроя")
        left_layout = QVBoxLayout()
        
        # Виджет визуализации
        self.visualization = RealTimeVisualization(
            parent=self,
            sheet_size=self.config.get_sheet_size_from_input(""),
            width=12, height=8, dpi=100
        )
        left_layout.addWidget(self.visualization)
        
        # Кнопки управления визуализацией
        vis_controls = QHBoxLayout()
        
        self.force_button = QPushButton("Показать силы")
        self.force_button.setCheckable(True)
        self.force_button.clicked.connect(lambda checked: self.visualization.toggle_forces(checked))
        vis_controls.addWidget(self.force_button)
        
        self.velocity_button = QPushButton("Показать скорости")
        self.velocity_button.setCheckable(True)
        self.velocity_button.clicked.connect(lambda checked: self.visualization.toggle_velocities(checked))
        vis_controls.addWidget(self.velocity_button)
        
        self.export_svg_button = QPushButton("Экспорт SVG")
        self.export_svg_button.clicked.connect(self.export_visualization_svg)
        vis_controls.addWidget(self.export_svg_button)
        
        left_layout.addLayout(vis_controls)
        left_panel.setLayout(left_layout)
        
        # Правая панель: управление и мониторинг
        right_panel = QSplitter(Qt.Vertical)
        
        # Верхняя часть: панель управления
        self.control_panel = ControlPanel()
        self.control_panel.start_optimization.connect(self.start_optimization)
        self.control_panel.stop_optimization.connect(self.stop_optimization)
        self.control_panel.pause_optimization.connect(self.pause_optimization)
        self.control_panel.reset_simulation.connect(self.reset_simulation)
        self.control_panel.export_results.connect(self.export_results)
        
        # Нижняя часть: панель мониторинга
        self.monitoring_panel = MonitoringPanel()
        
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.monitoring_panel)
        right_panel.setSizes([300, 600])
        
        # Добавление панелей в основной макет
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([1000, 600])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Меню
        self.create_menu()
        
        # Панель состояния
        self.statusBar().showMessage("Готов к работе")
        
        # Загрузка тестовых данных для демонстрации
        self.load_test_data()
    
    def create_menu(self):
        """Создание меню приложения"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu('Файл')
        
        load_action = file_menu.addAction('Загрузить данные...')
        load_action.triggered.connect(self.load_data)
        
        save_config_action = file_menu.addAction('Сохранить конфигурацию')
        save_config_action.triggered.connect(self.save_configuration)
        
        load_config_action = file_menu.addAction('Загрузить конфигурацию')
        load_config_action.triggered.connect(self.load_configuration)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Выход')
        exit_action.triggered.connect(self.close)
        
        # Меню Вид
        view_menu = menubar.addMenu('Вид')
        
        force_action = view_menu.addAction('Показать силы')
        force_action.setCheckable(True)
        force_action.triggered.connect(lambda checked: self.visualization.toggle_forces(checked))
        
        velocity_action = view_menu.addAction('Показать скорости')
        velocity_action.setCheckable(True)
        velocity_action.triggered.connect(lambda checked: self.visualization.toggle_velocities(checked))
        
        # Меню Помощь
        help_menu = menubar.addMenu('Помощь')
        
        about_action = help_menu.addAction('О программе')
        about_action.triggered.connect(self.show_about)
        
        docs_action = help_menu.addAction('Документация')
        docs_action.triggered.connect(self.show_documentation)
    
    def load_data(self):
        """Загрузка данных из файла"""
        dialog = LoadFileDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            filename = dialog.file_edit.text()
            sheet_size = (dialog.sheet_width.value(), dialog.sheet_height.value())
            tolerance = dialog.tolerance_spin.value()
            
            try:
                # Обновление конфигурации
                self.config.geometry['polygonization']['tolerance'] = tolerance
                self.constraint_manager.sheet_size = sheet_size
                self.visualization.sheet_size = sheet_size
                self.visualization.setup_plot()
                
                # Импорт данных в зависимости от формата
                if filename.endswith('.dxf'):
                    from my_io.dxf_import import import_dxf
                    self.current_shapes = import_dxf(filename, tolerance=tolerance)
                elif filename.endswith('.step') or filename.endswith('.stp'):
                    from my_io.step_import import import_step
                    self.current_shapes = import_step(filename, tolerance=tolerance, sheet_size=sheet_size)
                elif filename.endswith('.json'):
                    import json
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Здесь должна быть логика загрузки из JSON
                        QMessageBox.information(self, "Информация", "Поддержка JSON в разработке")
                        return
                
                # Обновление визуализации
                self.visualization.update_visualization([], self.current_defect_zones, {})
                
                # Обновление статуса
                self.statusBar().showMessage(f"Загружено {len(self.current_shapes)} фигур из {filename}")
                self.monitoring_panel.add_log_message(f"Загружено {len(self.current_shapes)} фигур")
                
                # Активация кнопок
                self.control_panel.enable_controls(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить данные: {str(e)}")
                self.monitoring_panel.add_log_message(f"Ошибка загрузки: {str(e)}")
    
    def load_test_data(self):
        """Загрузка тестовых данных для демонстрации"""
        from core.geometry import PolygonShape
        
        # Создание тестовых фигур
        shapes:list[PolygonShape] = []
        
        # Прямоугольные детали
        for i in range(8):
            width = np.random.uniform(50, 150)
            height = np.random.uniform(30, 100)
            contour = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            shape = PolygonShape(contour, name=f"rect_{i}")
            shape.priority = 1 if i < 4 else 2
            if i < 4:
                shape.allowed_angles = [0, 90]  # Ограничение на ориентацию
            shapes.append(shape)
        
        # L-образные детали
        for i in range(4):
            outer = [
                (0, 0), (100, 0), (100, 30), 
                (70, 30), (70, 100), (0, 100)
            ]
            shape = PolygonShape(outer, name=f"L_shape_{i}")
            shape.priority = 1
            shapes.append(shape)
        
        # Деталь с отверстием
        outer = [(0, 0), (120, 0), (120, 80), (0, 80)]
        hole = [(40, 20), (80, 20), (80, 60), (40, 60)]
        shape = PolygonShape(outer, [hole], name="part_with_hole")
        shape.priority = 1
        shapes.append(shape)
        
        # Дефектные зоны
        defect_zones = [
            PolygonShape([(950, 450), (1050, 450), (1050, 550), (950, 550)], name="defect_1"),
            PolygonShape([(1100, 600), (1200, 600), (1200, 700), (1100, 700)], name="defect_2")
        ]
        
        self.current_shapes = shapes
        self.current_defect_zones = [{'polygon': zone.polygon, 'type': 'general'} for zone in defect_zones]
        
        # Добавление дефектных зон в ConstraintManager
        for zone in defect_zones:
            self.constraint_manager.add_defect_zone(zone.outer_contour)
        
        # Обновление визуализации
        self.visualization.update_visualization([], self.current_defect_zones, {})
        
        self.statusBar().showMessage("Загружены тестовые данные")
        self.monitoring_panel.add_log_message("Загружены тестовые данные для демонстрации")
        
        # Активация кнопок
        self.control_panel.enable_controls(True)
    
    def start_optimization(self):
        """Запуск оптимизации"""
        if not self.current_shapes:
            QMessageBox.warning(self, "Предупреждение", "Нет загруженных фигур для оптимизации")
            return
        
        # Создание оптимизатора
        self.optimizer = PackingOptimizer({
            'sheet_size': self.constraint_manager.sheet_size,
            'min_gap': self.config.geometry['collision_detection']['min_gap'],
            'time_limit': self.control_panel.time_spin.value(),
            'use_sequential': self.control_panel.placement_mode == 'sequential',
            'stabilization_enabled': True
        })
        
        # Запуск потока оптимизации
        self.optimization_thread = OptimizationThread(
            self.optimizer, 
            self.current_shapes, 
            self.constraint_manager
        )
        self.optimization_thread.update_signal.connect(self.update_optimization)
        self.optimization_thread.finished_signal.connect(self.optimization_finished)
        self.optimization_thread.log_signal.connect(self.monitoring_panel.add_log_message)
        self.optimization_thread.error_signal.connect(self.handle_optimization_error)
        
        self.optimization_thread.start()
        
        # Обновление интерфейса
        self.control_panel.enable_controls(False)
        self.statusBar().showMessage("Оптимизация запущена...")
        self.monitoring_panel.add_log_message("Оптимизация запущена")
    
    def update_optimization(self, agents, defect_zones, forces, utilization, energy, iteration):
        """Обновление визуализации во время оптимизации"""
        self.visualization.update_visualization(agents, defect_zones, forces, self.constraint_manager)
        self.monitoring_panel.update_agents_table(agents)
        self.control_panel.update_progress(
            min(100, iteration),
            f"Итерация {iteration}",
            utilization,
            energy
        )
        self.statusBar().showMessage(f"Оптимизация: итерация {iteration}, использование {utilization:.1f}%")
    
    def optimization_finished(self, agents, utilization):
        """Завершение оптимизации"""
        self.current_agents = agents
        
        # Обновление интерфейса
        self.control_panel.enable_controls(True)
        self.control_panel.update_progress(100, "Оптимизация завершена", utilization, 0.0)
        self.statusBar().showMessage(f"Оптимизация завершена. Коэффициент использования: {utilization:.1f}%")
        self.monitoring_panel.add_log_message(f"Оптимизация завершена. η = {utilization:.1f}%")
        
        # Финальная визуализация
        self.visualization.update_visualization(agents, self.current_defect_zones, {})
    
    def stop_optimization(self):
        """Остановка оптимизации"""
        if hasattr(self, 'optimization_thread') and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.statusBar().showMessage("Оптимизация остановлена")
            self.monitoring_panel.add_log_message("Оптимизация остановлена пользователем")
    
    def pause_optimization(self):
        """Пауза оптимизации"""
        if hasattr(self, 'optimization_thread') and self.optimization_thread.isRunning():
            if self.optimization_thread.paused:
                self.optimization_thread.resume()
                self.control_panel.pause_button.setText("⏸ Пауза")
                self.statusBar().showMessage("Оптимизация возобновлена")
                self.monitoring_panel.add_log_message("Оптимизация возобновлена")
            else:
                self.optimization_thread.pause()
                self.control_panel.pause_button.setText("▶ Возобновить")
                self.statusBar().showMessage("Оптимизация приостановлена")
                self.monitoring_panel.add_log_message("Оптимизация приостановлена")
    
    def reset_simulation(self):
        """Сброс симуляции"""
        self.current_agents = []
        self.visualization.update_visualization([], self.current_defect_zones, {})
        self.monitoring_panel.clear_log()
        self.monitoring_panel.add_log_message("Симуляция сброшена")
        self.statusBar().showMessage("Симуляция сброшена")
        self.control_panel.update_progress(0, "Готов к запуску", 0.0, 0.0)
        self.control_panel.enable_controls(True)
    
    def export_results(self, format_type):
        """Экспорт результатов"""
        if not self.current_agents:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для экспорта")
            return
        
        # Формирование данных для экспорта
        placements = []
        for agent in self.current_agents:
            transformed_shape = agent.get_transformed_shape()
            placements.append({
                'shape': transformed_shape,
                'position': agent.position,
                'angle': agent.angle,
                'name': getattr(agent.shape, 'name', 'part'),
                'priority': agent.priority
            })
        
        # Выбор директории для экспорта
        output_dir = QFileDialog.getExistingDirectory(
            self, "Выберите директорию для экспорта", ""
        )
        
        if not output_dir:
            return
        
        try:
            # Экспорт (создаем подпапку с временной меткой)
            exported_files = export_results(
                placements=placements,
                output_dir=output_dir,
                sheet_size=self.constraint_manager.sheet_size,
                min_gap=self.config.geometry['collision_detection']['min_gap'],
                defect_zones=[PolygonShape(zone['polygon'].exterior.coords, name=f"defect_{i}") 
                            for i, zone in enumerate(self.current_defect_zones)],
                constraint_manager=self.constraint_manager,
                formats=[format_type],
                cutting_sequence=list(range(len(placements))),
                create_timestamped_folder=True  # Создавать подпапку с временной меткой
            )
            
            # Отображение результатов
            if format_type in exported_files:
                import os
                file_path = exported_files[format_type]
                dir_path = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                
                QMessageBox.information(
                    self, "Экспорт завершен", 
                    f"Результаты успешно экспортированы!\n\n"
                    f"Директория:\n{os.path.abspath(dir_path)}\n\n"
                    f"Файл:\n{file_name}"
                )
                self.monitoring_panel.add_log_message(f"Экспорт в {format_type}: {os.path.abspath(file_path)}")
            else:
                QMessageBox.warning(
                    self, "Экспорт не выполнен", 
                    f"Не удалось экспортировать результаты в формат {format_type}"
                )
        
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка экспорта", 
                f"Не удалось экспортировать результаты: {str(e)}"
            )
            self.monitoring_panel.add_log_message(f"Ошибка экспорта: {str(e)}")
    
    def export_visualization_svg(self):
        """Экспорт визуализации в SVG"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить визуализацию", "", "SVG Files (*.svg)"
        )
        
        if filename:
            if not filename.endswith('.svg'):
                filename += '.svg'
            
            self.visualization.export_to_svg(filename)
            self.monitoring_panel.add_log_message(f"Визуализация экспортирована в {filename}")
    
    def save_configuration(self):
        """Сохранение конфигурации"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить конфигурацию", "", "YAML Files (*.yaml);;JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.config.save_to_file(filename)
                QMessageBox.information(
                    self, "Конфигурация сохранена", 
                    f"Конфигурация успешно сохранена в:\n{filename}"
                )
                self.monitoring_panel.add_log_message(f"Конфигурация сохранена в {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка сохранения", 
                    f"Не удалось сохранить конфигурацию: {str(e)}"
                )
                self.monitoring_panel.add_log_message(f"Ошибка сохранения конфигурации: {str(e)}")
    
    def load_configuration(self):
        """Загрузка конфигурации"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить конфигурацию", "", "YAML Files (*.yaml);;JSON Files (*.json)"
        )
        
        if filename:
            try:
                self.config.load_from_file(filename)
                QMessageBox.information(
                    self, "Конфигурация загружена", 
                    f"Конфигурация успешно загружена из:\n{filename}"
                )
                self.monitoring_panel.add_log_message(f"Конфигурация загружена из {filename}")
                
                # Обновление интерфейса
                self.control_panel.profile_combo.setCurrentText(self.config.get_current_profile())
                self.control_panel.mode_combo.setCurrentText(self.config.placement['mode'])
                self.control_panel.tech_combo.setCurrentText(self.config.technological_constraints['cutting_technology'])
                self.control_panel.gap_spin.setValue(self.config.geometry['collision_detection']['min_gap'])
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка загрузки", 
                    f"Не удалось загрузить конфигурацию: {str(e)}"
                )
                self.monitoring_panel.add_log_message(f"Ошибка загрузки конфигурации: {str(e)}")
    
    def handle_optimization_error(self, error_message):
        """Обработка ошибок оптимизации"""
        QMessageBox.critical(self, "Ошибка оптимизации", error_message)
        self.statusBar().showMessage(f"Ошибка: {error_message}")
        self.control_panel.enable_controls(True)
    
    def show_about(self):
        """Показать информацию о программе"""
        QMessageBox.about(
            self, "О программе",
            "ИАГИ - Интеллектуальный агент гравитационной имитации\n"
            "для решения задач раскроя-упаковки плоских деталей\n\n"
            "Версия 1.0\n"
            "© 2025 Разработано в соответствии с требованиями\n"
            "ГОСТ 34029-2016 и ГОСТ Р ИСО 10303-21\n\n"
            "Алгоритмы основаны на теоретических моделях,\n"
            "описанных в главах 2-3 диссертации"
        )
    
    def show_documentation(self):
        """Показать документацию"""
        QMessageBox.information(
            self, "Документация",
            "Документация доступна в следующих источниках:\n\n"
            "1. Руководство пользователя.pdf\n"
            "2. Техническое описание API.pdf\n"
            "3. Методические рекомендации по настройке.pdf\n\n"
            "Полный пакет документации можно загрузить с сайта:\n"
            "https://example.com/iagi-docs"
        )
    
    def update_ui(self):
        """Обновление интерфейса"""
        # Этот метод вызывается таймером для обновления анимаций и состояния
        pass
    
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        if hasattr(self, 'optimization_thread') and self.optimization_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Подтверждение',
                "Оптимизация еще выполняется. Завершить приложение?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.optimization_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    
    # Настройка стиля приложения
    app.setStyle('Fusion')
    
    # Создание градиента для фона
    palette = QPalette()
    gradient = QLinearGradient(0, 0, 0, 400)
    gradient.setColorAt(0.0, QColor(45, 45, 60))
    gradient.setColorAt(1.0, QColor(30, 30, 40))
    palette.setBrush(QPalette.Window, QBrush(gradient))
    app.setPalette(palette)
    
    # Создание и отображение главного окна
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()