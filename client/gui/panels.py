#region Imports
from typing import Dict, Any, Optional, List
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QComboBox, QLabel, QDoubleSpinBox, QSpinBox,
    QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox, QFrame, 
)
from PySide6.QtCore import (
    Signal as pyqtSignal
)
from PySide6.QtGui import QFont, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#endregion

class ControlPanel(QWidget):
    """Панель управления процессом раскроя ИАГИ"""
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
        
        self.overlay = QFrame(self)
        self.overlay.setStyleSheet("background-color: rgba(45, 45, 45, 180); border-radius: 4px;")
        self.overlay.setVisible(False)
        self.overlay.raise_() # Поднимаем поверх всех элементов
        
        self.overlay_label = QLabel("🔒 Панель заблокирована\nИдёт процесс оптимизации", self.overlay)
        self.overlay_label.setStyleSheet("color: #FFFFFF; font-size: 14px; font-weight: bold;")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.raise_()
        
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
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Технологические параметры
        tech_group = QGroupBox("Технологические параметры")
        tech_layout = QVBoxLayout()
        self.tech_combo = QComboBox()
        self.tech_combo.addItems(['laser', 'plasma', 'waterjet', 'default'])
        self.tech_combo.setCurrentText(self.technology)
        self.tech_combo.currentTextChanged.connect(self.on_technology_changed)
        tech_layout.addWidget(QLabel("Технология резки:"))
        tech_layout.addWidget(self.tech_combo)
        
        gap_layout = QHBoxLayout()
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.1, 10.0); self.gap_spin.setValue(0.5); self.gap_spin.setSingleStep(0.1)
        gap_layout.addWidget(QLabel("Минимальный зазор (мм):")); gap_layout.addWidget(self.gap_spin)
        tech_layout.addLayout(gap_layout)
        
        time_layout = QHBoxLayout()
        self.time_spin = QSpinBox()
        self.time_spin.setRange(10, 3600); self.time_spin.setValue(300); self.time_spin.setSingleStep(30)
        time_layout.addWidget(QLabel("Макс. время (сек):")); time_layout.addWidget(self.time_spin)
        tech_layout.addLayout(time_layout)
        tech_group.setLayout(tech_layout)
        layout.addWidget(tech_group)
        
        # Размеры листа
        sheet_group = QGroupBox("Размеры листа")
        sheet_layout = QVBoxLayout()
        width_layout = QHBoxLayout()
        self.sheet_width_spin = QDoubleSpinBox()
        self.sheet_width_spin.setRange(100, 10000); self.sheet_width_spin.setValue(2000); self.sheet_width_spin.setSingleStep(100)
        width_layout.addWidget(QLabel("Ширина (мм):")); width_layout.addWidget(self.sheet_width_spin)
        sheet_layout.addLayout(width_layout)
        
        height_layout = QHBoxLayout()
        self.sheet_height_spin = QDoubleSpinBox()
        self.sheet_height_spin.setRange(100, 10000); self.sheet_height_spin.setValue(1000); self.sheet_height_spin.setSingleStep(100)
        height_layout.addWidget(QLabel("Высота (мм):")); height_layout.addWidget(self.sheet_height_spin)
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
        
        # Исправление: добавлены отсутствующие в оригинале кнопки
        self.pause_button = QPushButton("⏸ Пауза"); self.pause_button.setEnabled(False)
        button_layout.addWidget(self.pause_button)
        self.reset_button = QPushButton("🔄 Сброс"); self.reset_button.setEnabled(False)
        button_layout.addWidget(self.reset_button)
        self.stop_button = QPushButton("⏹ Остановить"); self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        self.progress_label = QLabel("Готов к работе")
        control_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        control_layout.addWidget(self.progress_bar)
        
        stats_layout = QHBoxLayout()
        self.utilization_label = QLabel("Использование: -")
        stats_layout.addWidget(self.utilization_label)
        self.energy_label = QLabel("Энергия: 0.0") # Исправление: добавлена отсутствующая метка
        stats_layout.addWidget(self.energy_label)
        control_layout.addLayout(stats_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Экспорт
        export_group = QGroupBox("Экспорт результатов")
        export_layout = QHBoxLayout()
        self.dxf_button = QPushButton("DXF"); self.dxf_button.setEnabled(False); export_layout.addWidget(self.dxf_button)
        self.svg_button = QPushButton("SVG"); self.svg_button.setEnabled(False); export_layout.addWidget(self.svg_button)
        self.json_button = QPushButton("JSON"); self.json_button.setEnabled(False); export_layout.addWidget(self.json_button)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        self.setLayout(layout)

    def resizeEvent(self, event):
        """Гарантируем, что оверлей всегда покрывает всю панель"""
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.overlay_label.setGeometry(self.rect())

    #region Handlers & State Management
    def on_profile_changed(self, profile_name):
        self.current_profile = profile_name
        QMessageBox.information(self, "Профиль изменен", f"Применен профиль: {profile_name}")

    def on_mode_changed(self, mode):
        self.placement_mode = mode
        if mode == 'hybrid':
            QMessageBox.information(self, "Гибридный режим", "Включены гибридные алгоритмы (ГА-ИАГИ, PSO-ИАГИ).")

    def on_technology_changed(self, technology):
        self.technology = technology
        QMessageBox.information(self, "Технология изменена", f"Установлена технология: {technology}")

    def enable_controls(self, enable: bool):
        self.start_button.setEnabled(enable)
        self.pause_button.setEnabled(not enable)
        self.stop_button.setEnabled(not enable)
        self.reset_button.setEnabled(enable) # Кнопка сброса всегда доступна, если не идет процесс, или можно сделать её доступной всегда
        self.profile_combo.setEnabled(enable)
        self.mode_combo.setEnabled(enable)
        self.tech_combo.setEnabled(enable)
        self.gap_spin.setEnabled(enable)
        self.time_spin.setEnabled(enable)
        
        # Управление оверлеем
        #self.overlay.setVisible(not enable)

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
    #endregion


class MonitoringPanel(QWidget):
    """Панель мониторинга процесса оптимизации ИАГИ (Объединенная)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        self.history_iterations = []
        self.history_energies = []
        self.history_utilizations = []

    def init_ui(self):
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Вкладка: Статистика
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        self.tasks_table = QTableWidget()
        self.tasks_table.setColumnCount(4)
        self.tasks_table.setHorizontalHeaderLabels(["ID задачи", "Статус", "Прогресс", "Сообщение"])
        self.tasks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stats_layout.addWidget(QLabel("Текущая задача:"))
        stats_layout.addWidget(self.tasks_table)
        
        self.agents_table = QTableWidget()
        self.agents_table.setColumnCount(7)
        self.agents_table.setHorizontalHeaderLabels(["ID", "Имя", "Приоритет", "Позиция", "Угол", "Скорость", "Статус"])
        self.agents_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stats_layout.addWidget(QLabel("Состояние фигур:"))
        stats_layout.addWidget(self.agents_table)
        
        energy_layout = QHBoxLayout()
        self.total_energy_label = QLabel("Полная энергия: 0.0")
        self.kinetic_energy_label = QLabel("Кинетическая: 0.0")
        self.potential_energy_label = QLabel("Потенциальная: 0.0")
        energy_layout.addWidget(self.total_energy_label)
        energy_layout.addWidget(self.kinetic_energy_label)
        energy_layout.addWidget(self.potential_energy_label)
        stats_layout.addLayout(energy_layout)
        
        stats_tab.setLayout(stats_layout)
        self.tabs.addTab(stats_tab, "Статистика")
        
        # Вкладка: Графики
        graphs_tab = QWidget()
        graphs_layout = QVBoxLayout()
        
        self.energy_fig = Figure(figsize=(5, 3), dpi=100)
        self.energy_canvas = FigureCanvas(self.energy_fig)
        self.energy_ax = self.energy_fig.add_subplot(111)
        self.energy_ax.set_title('Диссипация энергии системы', fontsize=10)
        self.energy_ax.set_xlabel('Итерация')
        self.energy_ax.set_ylabel('Энергия (усл. ед.)')
        self.energy_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Инициализация линий (один раз!) для обновления через set_data
        self.energy_line, = self.energy_ax.plot([], [], 'b-', linewidth=2, label='Полная энергия')
        self.anomaly_line, = self.energy_ax.plot([], [], 'ro', markersize=6, label='Аномалия')
        self.energy_ax.legend(loc='upper right', fontsize=8)
        
        graphs_layout.addWidget(self.energy_canvas)
        
        self.utilization_fig = Figure(figsize=(5, 3), dpi=100)
        self.utilization_canvas = FigureCanvas(self.utilization_fig)
        self.utilization_ax = self.utilization_fig.add_subplot(111)
        self.utilization_ax.set_title('Сходимость к оптимальной плотности', fontsize=10)
        self.utilization_ax.set_xlabel('Итерация')
        self.utilization_ax.set_ylabel('Использование материала (%)')
        self.utilization_ax.set_ylim(0, 100)
        self.utilization_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Инициализация линии (один раз!)
        self.utilization_line, = self.utilization_ax.plot([], [], 'g-', linewidth=2, label='η (коэффициент использования)')
        self.utilization_ax.legend(loc='lower right', fontsize=8)
        
        graphs_layout.addWidget(self.utilization_canvas)
        
        graphs_tab.setLayout(graphs_layout)
        self.tabs.addTab(graphs_tab, "Графики")
        
        # Вкладка: Лог
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))
        log_layout.addWidget(QLabel("Лог событий:"))
        log_layout.addWidget(self.log_text)
        
        clear_button = QPushButton("Очистить лог")
        clear_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_button)
        
        log_tab.setLayout(log_layout)
        self.tabs.addTab(log_tab, "Лог")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    #region Monitoring Updates
    def update_task_status(self, task_id: str, status: str, progress: int, message: str):
        self.tasks_table.setRowCount(1)
        self.tasks_table.setItem(0, 0, QTableWidgetItem(task_id))
        self.tasks_table.setItem(0, 1, QTableWidgetItem(status))
        self.tasks_table.setItem(0, 2, QTableWidgetItem(f"{progress}%"))
        self.tasks_table.setItem(0, 3, QTableWidgetItem(message))
        
    def update_agents_table(self, shapes_data: List[Dict]):
        """
        Обновление таблицы агентов из current_shapes.
        
        Формат shapes_data (согласно S-2 сервера):
            - id: int
            - name: str
            - contour: List[List[float]]
            - centroid: [x, y]
            - angle: float
            - priority: int
            - is_frozen: bool
            - velocity: [vx, vy]
            - angular_velocity: float
        """
        
        print(f"Передача в update_agents_table shapes_data: {shapes_data}")
        
        if not shapes_data:
            self.agents_table.setRowCount(0)
            print("shapes_data пустой")
            return
        
        # Отключаем обновление для массового заполнения (без мигания)
        self.agents_table.setUpdatesEnabled(False)
        
        try:
            self.agents_table.setRowCount(len(shapes_data))
            for row, shape in enumerate(shapes_data):
                # Колонка 0: ID
                self.agents_table.setItem(
                    row, 0, QTableWidgetItem(str(shape.get('id', row)))
                )
                
                # Колонка 1: Имя
                name = shape.get('name') or f"part_{shape.get('id', row) + 1}"
                self.agents_table.setItem(row, 1, QTableWidgetItem(name))
                
                # Колонка 2: Приоритет
                self.agents_table.setItem(
                    row, 2, QTableWidgetItem(str(shape.get('priority', 2)))
                )
                
                # Колонка 3: Позиция (centroid)
                centroid = shape.get('centroid') or [0, 0]
                if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                    cx, cy = centroid[0], centroid[1]
                else:
                    cx, cy = 0.0, 0.0
                self.agents_table.setItem(
                    row, 3, QTableWidgetItem(f"({cx:.1f}, {cy:.1f})")
                )
                
                # Колонка 4: Угол
                angle = shape.get('angle', 0.0) or 0.0
                self.agents_table.setItem(
                    row, 4, QTableWidgetItem(f"{angle:.1f}°")
                )
                
                # Колонка 5: Скорость (норма вектора ‖v‖ = √(vx² + vy²))
                velocity = shape.get('velocity') or [0, 0]
                if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
                    v_norm = (velocity[0]**2 + velocity[1]**2) ** 0.5
                else:
                    v_norm = 0.0
                self.agents_table.setItem(
                    row, 5, QTableWidgetItem(f"{v_norm:.2f}")
                )
                
                # Колонка 6: Статус (с иконкой 🔒 для frozen)
                is_frozen = shape.get('is_frozen', False)
                status_text = "🔒 frozen" if is_frozen else "▶ active"
                self.agents_table.setItem(row, 6, QTableWidgetItem(status_text))
        finally:
            self.agents_table.setUpdatesEnabled(True)
            
    def reset_history(self):
        """Сброс локальной истории (вызывается при reset_task)"""
        self.history_iterations = []
        self.history_energies = []
        self.history_utilizations = []
    
    def update_energy_graphs(self, iterations: List, energies: List[float], utilizations: List[float]):
        """
        Обновление графиков энергии и утилизации БЕЗ МИГАНИЯ.
        Согласно Главе 2.6.3 диссертации (Теорема 1), энергия должна
        монотонно не возрастать (диссипативность системы).
        
        C-FIX-3: Накапливаем историю локально, сервер передаёт только дельту.
        """
        if not iterations:
            return
        
        # ← НОВОЕ (C-FIX-3): Накапливаем историю локально
        self.history_iterations.extend(iterations)
        self.history_energies.extend(energies)
        self.history_utilizations.extend(utilizations)
        
        # Ограничиваем длину истории (последние 1000 точек для производительности)
        max_points = 1000
        if len(self.history_iterations) > max_points:
            self.history_iterations = self.history_iterations[-max_points:]
            self.history_energies = self.history_energies[-max_points:]
            self.history_utilizations = self.history_utilizations[-max_points:]
        
        # === 1. График энергии (set_data вместо clear+plot) ===
        self.energy_line.set_data(self.history_iterations, self.history_energies)
        
        # Проверка монотонности: если энергия выросла >1% — это аномалия
        # (допустимы только численные флуктуации <1%)
        anomalies_x, anomalies_y = [], []
        for i in range(1, len(self.history_energies)):
            if self.history_energies[i] > self.history_energies[i-1] * 1.01:
                anomalies_x.append(self.history_iterations[i])
                anomalies_y.append(self.history_energies[i])
        
        self.anomaly_line.set_data(anomalies_x, anomalies_y)
        
        # Авто-масштабирование осей
        self.energy_ax.relim()
        self.energy_ax.autoscale_view()
        
        # Перерисовка БЕЗ МИГАНИЯ (draw_idle вместо draw)
        self.energy_canvas.draw_idle()
        
        # === 2. График утилизации (set_data вместо clear+plot) ===
        self.utilization_line.set_data(self.history_iterations, self.history_utilizations)
        
        # Авто-масштаб только по X, Y фиксирован [0, 100]
        self.utilization_ax.relim()
        self.utilization_ax.autoscale_view()
        self.utilization_canvas.draw_idle()
       
    def update_energy_stats(self, total_energy: float, kinetic_energy: float, potential_energy: float):
        self.total_energy_label.setText(f"Полная энергия: {total_energy:.2f}")
        self.kinetic_energy_label.setText(f"Кинетическая: {kinetic_energy:.2f}")
        self.potential_energy_label.setText(f"Потенциальная: {potential_energy:.2f}")
    
    def reset_graphs(self):
        """Сброс графиков к начальному состоянию (вызывается при reset_task)"""
        
        self.reset_history()
        
        self.energy_line.set_data([], [])
        self.anomaly_line.set_data([], [])
        self.energy_ax.relim()
        self.energy_ax.autoscale_view()
        self.energy_canvas.draw_idle()
        
        self.utilization_line.set_data([], [])
        self.utilization_ax.relim()
        self.utilization_ax.autoscale_view(axis='x')
        self.utilization_canvas.draw_idle()
    #endregion

    #region Log Management
    def add_log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()
    #endregion