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
        self.agents_table.setColumnCount(6)
        self.agents_table.setHorizontalHeaderLabels(["ID", "Имя", "Приоритет", "Положение", "Угол", "Статус"])
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
        self.energy_ax.set_title('Изменение энергии системы'); self.energy_ax.grid(True)
        graphs_layout.addWidget(self.energy_canvas)
        
        self.utilization_fig = Figure(figsize=(5, 3), dpi=100)
        self.utilization_canvas = FigureCanvas(self.utilization_fig)
        self.utilization_ax = self.utilization_fig.add_subplot(111)
        self.utilization_ax.set_title('Коэффициент использования'); self.utilization_ax.grid(True)
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
        if not shapes_data:
            self.agents_table.setRowCount(0)
            return
            
        self.agents_table.setRowCount(len(shapes_data))
        for i, shape in enumerate(shapes_data):
            self.agents_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.agents_table.setItem(i, 1, QTableWidgetItem(shape.get('name', f'Part_{i+1}')))
            self.agents_table.setItem(i, 2, QTableWidgetItem(str(shape.get('priority', 2))))
            
            centroid = shape.get('centroid', (0, 0))
            # Обработка случая, когда centroid может быть списком или кортежем
            cx = centroid[0] if isinstance(centroid, (list, tuple)) else 0
            cy = centroid[1] if isinstance(centroid, (list, tuple)) else 0
            self.agents_table.setItem(i, 3, QTableWidgetItem(f"({cx:.1f}, {cy:.1f})"))
            
            self.agents_table.setItem(i, 4, QTableWidgetItem(f"{shape.get('angle', 0):.1f}°"))
            self.agents_table.setItem(i, 5, QTableWidgetItem("Активна" if not shape.get('is_frozen', False) else "Заморожена"))

    def update_energy_graphs(self, iterations: List, energies: List[float], utilizations: List[float]):
        if not iterations or len(iterations) < 2:
            return
        # 1. График энергии (согласно Главе 2.6.3, энергия должна монотонно убывать)
        self.energy_ax.clear()
        self.energy_ax.plot(iterations, energies, 'b-', linewidth=2, label='Полная энергия системы')
        self.energy_ax.set_title('Диссипация энергии системы', fontsize=10)
        self.energy_ax.set_xlabel('Итерация / Время (с)')
        self.energy_ax.set_ylabel('Энергия (усл. ед.)')
        self.energy_ax.legend(loc='upper right')
        self.energy_ax.grid(True, linestyle='--', alpha=0.7)
        self.energy_canvas.draw()
        
        # 2. График использования (Коэффициент $\eta$ должен стремиться к максимуму)
        self.utilization_ax.clear()
        self.utilization_ax.plot(iterations, utilizations, 'g-', linewidth=2, label='Коэффициент использования ($\eta$)')
        self.utilization_ax.set_title('Сходимость к оптимальной плотности', fontsize=10)
        self.utilization_ax.set_xlabel('Итерация / Время (с)')
        self.utilization_ax.set_ylabel('Использование материала (%)')
        self.utilization_ax.set_ylim(0, 100) # Фиксируем ось для наглядности
        self.utilization_ax.legend(loc='lower right')
        self.utilization_ax.grid(True, linestyle='--', alpha=0.7)
        self.utilization_canvas.draw()

    def update_energy_stats(self, total_energy: float, kinetic_energy: float, potential_energy: float):
        self.total_energy_label.setText(f"Полная энергия: {total_energy:.2f}")
        self.kinetic_energy_label.setText(f"Кинетическая: {kinetic_energy:.2f}")
        self.potential_energy_label.setText(f"Потенциальная: {potential_energy:.2f}")
    #endregion

    #region Log Management
    def add_log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()
    #endregion