"""
Клиентская часть приложения ИАГИ на PyQt5 с HTTP-клиентом
Подключается к серверу FastAPI для выполнения оптимизации раскроя
"""

import sys
import time
import threading
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QTabWidget, QFileDialog, QProgressBar, QMessageBox,
                             QTextEdit, QSplitter, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QDialog, QFormLayout, QDialogButtonBox, QLineEdit,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QUrl
from PyQt5.QtGui import QColor, QPalette, QLinearGradient, QBrush, QPainter, QPen, QPolygonF, QFont

# Добавляем корень проекта в путь импорта
sys.path.append(str(Path(__file__).parent.parent))

from core.geometry import PolygonShape
from config.settings import get_config


class ServerConnectionDialog(QDialog):
    """Диалог настройки подключения к серверу"""
    
    def __init__(self, parent=None, current_url: str = "http://localhost:8000"):
        super().__init__(parent)
        self.setWindowTitle("Настройка подключения к серверу")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.current_url = current_url
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


class APIClient:
    """HTTP клиент для взаимодействия с сервером ИАГИ"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = 300  # 5 минут таймаут
    
    def check_connection(self) -> bool:
        """Проверка подключения к серверу"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Получение информации о сервере"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Не удалось получить информацию о сервере: {e}")
    
    def get_profiles(self) -> List[Dict[str, Any]]:
        """Получение доступных профилей"""
        try:
            response = self.session.get(f"{self.base_url}/api/profiles", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('profiles', [])
        except Exception as e:
            raise ConnectionError(f"Не удалось получить профили: {e}")
    
    def get_algorithms(self) -> List[Dict[str, Any]]:
        """Получение доступных алгоритмов"""
        try:
            response = self.session.get(f"{self.base_url}/api/algorithms", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('algorithms', [])
        except Exception as e:
            raise ConnectionError(f"Не удалось получить алгоритмы: {e}")
    
    def get_shapes_info(self, file_path: str) -> Dict[str, Any]:
        """Получение информации о фигурах в файле"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f)}
                response = self.session.post(
                    f"{self.base_url}/api/shapes/info",
                    files=files,
                    timeout=30
                )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Не удалось получить информацию о фигурах: {e}")
    
    def start_optimization(
        self,
        file_path: str,
        profile: str = 'medium_precision',
        algorithm: str = 'sequential',
        technology: str = 'laser',
        min_gap: float = 0.5,
        time_limit: int = 300,
        sheet_width: float = 2000.0,
        sheet_height: float = 1000.0,
        use_original_positions: bool = True
    ) -> str:
        """Запуск оптимизации и возврат task_id"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f)}
                data = {
                    'profile': profile,
                    'algorithm': algorithm,
                    'technology': technology,
                    'min_gap': min_gap,
                    'time_limit': time_limit,
                    'sheet_width': sheet_width,
                    'sheet_height': sheet_height,
                    'use_original_positions': use_original_positions
                }
                response = self.session.post(
                    f"{self.base_url}/api/optimize",
                    files=files,
                    data=data,
                    timeout=30
                )
            response.raise_for_status()
            result = response.json()
            return result['task_id']
        except Exception as e:
            raise ConnectionError(f"Не удалось запустить оптимизацию: {e}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Получение статуса задачи"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/status/{task_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Не удалось получить статус задачи: {e}")
    
    def download_result(self, task_id: str, format_type: str, output_path: str):
        """Скачивание результата оптимизации"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/download/{task_id}/{format_type}",
                timeout=60
            )
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
        except Exception as e:
            raise ConnectionError(f"Не удалось скачать результат: {e}")


class OptimizationThread(QThread):
    """Поток для мониторинга задачи оптимизации"""
    
    status_update = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, api_client: APIClient, task_id: str, poll_interval: float = 1.0):
        super().__init__()
        self.api_client = api_client
        self.task_id = task_id
        self.poll_interval = poll_interval
        self.running = True
    
    def run(self):
        """Мониторинг статуса задачи"""
        try:
            while self.running:
                status = self.api_client.get_task_status(self.task_id)
                self.status_update.emit(status)
                
                if status['status'] in ['completed', 'failed']:
                    self.finished_signal.emit(status)
                    break
                
                time.sleep(self.poll_interval)
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def stop(self):
        self.running = False


class ControlPanel(QWidget):
    """Панель управления клиентом"""
    
    start_optimization = pyqtSignal()
    stop_optimization = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_config()
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
        profile_layout.addWidget(QLabel("Выберите профиль:"))
        profile_layout.addWidget(self.profile_combo)
        
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)
        
        # Режим размещения
        mode_group = QGroupBox("Режим размещения")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['sequential', 'parallel', 'hybrid'])
        self.mode_combo.setCurrentText(self.placement_mode)
        mode_layout.addWidget(QLabel("Выберите режим:"))
        mode_layout.addWidget(self.mode_combo)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Технологические параметры
        tech_group = QGroupBox("Технологические параметры")
        tech_layout = QVBoxLayout()
        
        tech_layout.addWidget(QLabel("Технология резки:"))
        self.tech_combo = QComboBox()
        self.tech_combo.addItems(['laser', 'plasma', 'waterjet', 'default'])
        self.tech_combo.setCurrentText(self.technology)
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
        
        self.start_button = QPushButton("▶ Запустить оптимизацию")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Остановить")
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
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
    
    def enable_controls(self, enable: bool):
        self.start_button.setEnabled(enable)
        self.stop_button.setEnabled(not enable)
        self.profile_combo.setEnabled(enable)
        self.mode_combo.setEnabled(enable)
        self.tech_combo.setEnabled(enable)
        self.gap_spin.setEnabled(enable)
        self.time_spin.setEnabled(enable)
    
    def enable_export(self, enable: bool):
        self.dxf_button.setEnabled(enable)
        self.svg_button.setEnabled(enable)
        self.json_button.setEnabled(enable)
    
    def update_progress(self, progress: int, status: str, utilization: Optional[float] = None):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)
        if utilization is not None:
            self.utilization_label.setText(f"Использование: {utilization:.1f}%")


class MainWindow(QMainWindow):
    """Главное окно клиентского приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ИАГИ Клиент - Раскрой плоских деталей")
        self.setGeometry(100, 100, 1200, 800)
        
        # Настройки подключения
        self.server_url = "http://localhost:8000"
        self.api_client = APIClient(self.server_url)
        
        # Текущие данные
        self.current_file = None
        self.current_task_id = None
        self.optimization_thread: Optional[OptimizationThread] = None
        
        self.init_ui()
        self.check_server_connection()
    
    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Верхняя панель с настройками подключения
        top_panel = QHBoxLayout()
        
        self.connection_label = QLabel("Сервер: Не подключено")
        self.connection_label.setStyleSheet("color: red; font-weight: bold;")
        top_panel.addWidget(self.connection_label)
        
        self.connect_button = QPushButton("🔌 Подключение...")
        self.connect_button.clicked.connect(self.show_connection_dialog)
        top_panel.addWidget(self.connect_button)
        
        self.refresh_button = QPushButton("🔄 Обновить")
        self.refresh_button.clicked.connect(self.check_server_connection)
        top_panel.addWidget(self.refresh_button)
        
        top_panel.addStretch()
        
        main_layout.addLayout(top_panel)
        
        # Основная область
        splitter = QSplitter(Qt.Horizontal)
        
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
        
        left_panel.setLayout(left_layout)
        
        # Правая панель: лог и информация
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        log_group = QGroupBox("Журнал событий")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        info_group = QGroupBox("Информация о задаче")
        info_layout = QVBoxLayout()
        
        self.task_info = QTextEdit()
        self.task_info.setReadOnly(True)
        self.task_info.setMaximumHeight(200)
        info_layout.addWidget(self.task_info)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 500])
        
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
        
        help_menu = menubar.addMenu('Помощь')
        
        about_action = help_menu.addAction('О программе')
        about_action.triggered.connect(self.show_about)
    
    def show_connection_dialog(self):
        dialog = ServerConnectionDialog(self, self.server_url)
        if dialog.exec_() == QDialog.Accepted:
            self.server_url = dialog.get_server_url()
            self.api_client = APIClient(self.server_url)
            self.check_server_connection()
    
    def check_server_connection(self):
        if self.api_client.check_connection():
            self.connection_label.setText(f"Сервер: {self.server_url} ✓")
            self.connection_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_message("Подключение к серверу успешно")
            self.statusBar().showMessage(f"Подключено к {self.server_url}")
        else:
            self.connection_label.setText("Сервер: Не подключено")
            self.connection_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_message("Не удалось подключиться к серверу")
            self.statusBar().showMessage("Сервер недоступен")
    
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
            self.log_message(f"Выбран файл: {filename}")
            
            # Получение информации о фигурах
            try:
                shapes_info = self.api_client.get_shapes_info(filename)
                num_shapes = shapes_info.get('num_shapes', 0)
                self.shapes_info_label.setText(f"{num_shapes} фигур")
                self.control_panel.start_button.setEnabled(True)
                self.log_message(f"Загружено {num_shapes} фигур")
            except Exception as e:
                self.shapes_info_label.setText("Ошибка")
                self.log_message(f"Ошибка получения информации: {e}")
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
            self.log_message(f"Задача запущена: {task_id}")
            self.task_info.setText(f"ID задачи: {task_id}\nСтатус: В ожидании...")
            
            # Запуск мониторинга
            self.optimization_thread = OptimizationThread(self.api_client, task_id)
            self.optimization_thread.status_update.connect(self.on_status_update)
            self.optimization_thread.finished_signal.connect(self.on_optimization_finished)
            self.optimization_thread.error_signal.connect(self.on_optimization_error)
            self.optimization_thread.start()
            
            self.control_panel.stop_button.setEnabled(True)
            self.statusBar().showMessage("Оптимизация запущена...")
            
        except Exception as e:
            self.log_message(f"Ошибка запуска: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить оптимизацию: {e}")
            self.control_panel.enable_controls(True)
    
    def on_status_update(self, status: dict):
        self.control_panel.update_progress(
            status.get('progress', 0),
            status.get('message', ''),
            status.get('utilization')
        )
        
        self.task_info.setText(
            f"ID задачи: {self.current_task_id}\n"
            f"Статус: {status.get('status', 'unknown')}\n"
            f"Прогресс: {status.get('progress', 0)}%\n"
            f"Сообщение: {status.get('message', '')}"
        )
    
    def on_optimization_finished(self, status: dict):
        self.log_message(f"Оптимизация завершена: {status.get('message', '')}")
        self.control_panel.enable_export(True)
        self.control_panel.stop_button.setEnabled(False)
        self.statusBar().showMessage(f"Оптимизация завершена. Использование: {status.get('utilization', 0):.1f}%")
    
    def on_optimization_error(self, error: str):
        self.log_message(f"Ошибка: {error}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка оптимизации: {error}")
        self.control_panel.enable_controls(True)
    
    def stop_optimization(self):
        if self.optimization_thread:
            self.optimization_thread.stop()
            self.log_message("Оптимизация остановлена пользователем")
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
            self.log_message(f"Результат сохранен: {output_path}")
            QMessageBox.information(self, "Экспорт завершен", f"Файл сохранен:\n{output_path}")
        except Exception as e:
            self.log_message(f"Ошибка экспорта: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результат: {e}")
    
    def log_message(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def show_about(self):
        QMessageBox.about(
            self, "О программе",
            "ИАГИ Клиент\n\n"
            "Клиентское приложение для системы оптимизации раскроя\n"
            "плоских деталей с использованием гравитационной имитации.\n\n"
            "Версия 1.0\n"
            "Подключается к серверу ИАГИ через HTTP API"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    palette = QPalette()
    gradient = QLinearGradient(0, 0, 0, 400)
    gradient.setColorAt(0.0, QColor(45, 45, 60))
    gradient.setColorAt(1.0, QColor(30, 30, 40))
    palette.setBrush(QPalette.Window, QBrush(gradient))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
