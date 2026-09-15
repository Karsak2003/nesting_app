#region Imports
from pathlib import Path
from typing import Optional, Any
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QGroupBox, QLabel, QPushButton, QTextEdit, QFileDialog,
                             QMessageBox, QMenuBar, QDialog)
from PySide6.QtCore import QSettings
from .config import get_client_config
from .visualization import RealTimeVisualization
from .dialogs import ServerConnectionDialog
from .panels import ControlPanel, MonitoringPanel
#endregion

class MainWindow(QMainWindow):
    """Главное окно клиентского приложения ИАГИ"""
    def __init__(self, server_url: str = "http://localhost:8000", theme: str = "system"):
        super().__init__()
        self.setWindowTitle("ИАГИ Клиент - Раскрой плоских деталей")
        self.setGeometry(100, 100, 1400, 900)
        
        self.settings = QSettings("IAGI", "Client")
        self.server_url = server_url or self.settings.value("server_url", "http://localhost:8000")
        self.current_theme = theme or self.settings.value("theme", "system")
        
        self.last_iteration = 0
        # Импорт APIClient здесь, чтобы избежать циклических зависимостей
        try:
            from .api_client import APIClient, OptimizationThread
            self.api_client = APIClient(self.server_url)
            self.OptimizationThread = OptimizationThread
        except ImportError:
            self.api_client = None
            self.OptimizationThread = None
            print("Предупреждение: Модуль client.main не найден. Запуск в демо-режиме.")
        
        self.current_file = None
        self.current_task_id = None
        self.optimization_thread: Optional[Any] = None
        self.current_result_data = None
        
        self.init_ui()
        self.apply_theme(self.current_theme)
        self.check_server_connection()
        self.control_panel.reset_button.clicked.connect(self.reset_task)

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Левая панель: управление
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
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
        self.visualization = RealTimeVisualization(parent=self, sheet_size=(2000, 1000), width=12, height=8, dpi=100)
        center_layout.addWidget(self.visualization)
        
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
        
        # Сборка макета
        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 700, 350])
        main_layout.addWidget(splitter)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        self.create_menu()
        self.statusBar().showMessage("Готов к работе")

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Файл')
        connect_action = file_menu.addAction('Подключение к серверу...')
        connect_action.triggered.connect(self.show_connection_dialog)
        file_menu.addSeparator()
        exit_action = file_menu.addAction('Выход')
        exit_action.triggered.connect(self.close)
        
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

    def reset_task(self):
        """Полный сброс задачи без перезапуска ПО (Исправление п.5)"""
        # 1. Останавливаем поток, если он работает
        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.stop()
            self.optimization_thread.wait(1000) # Ждем завершения
            
        # 2. Очищаем внутренние данные
        self.current_task_id = None
        self.current_result_data = None
        self.last_iteration = 0 
        
        # 3. Разблокируем интерфейс
        self.control_panel.enable_controls(True)
        self.control_panel.enable_export(False)
        
        # 4. Очищаем мониторинг
        self.monitoring_panel.tasks_table.setRowCount(0)
        self.monitoring_panel.agents_table.setRowCount(0)
        self.monitoring_panel.clear_log()

        # 5. Очищаем графики (атомарный сброс без пересоздания осей)
        self.monitoring_panel.reset_graphs()
        
        # 6. Очищаем визуализацию
        self.visualization.reset() 
        
        self.log_message("Задача сброшена. Система готова к новой оптимизации.")
        self.statusBar().showMessage("Готов к работе")
    
    #region Theme & Connection
    def apply_theme(self, theme: str):
        self.current_theme = theme
        self.settings.setValue("theme", theme)
        try:
            from .themes import get_palette, generate_stylesheet
            palette = get_palette(theme)
            stylesheet = generate_stylesheet(palette)
            self.setStyleSheet(stylesheet)
        except ImportError:
            pass # Игнорируем, если темы нет в демо-режиме
            
        if hasattr(self, 'theme_system_action'):
            self.theme_system_action.setChecked(theme == 'system')
            self.theme_light_action.setChecked(theme == 'light')
            self.theme_dark_action.setChecked(theme == 'dark')
        self.monitoring_panel.add_log_message(f"Применена тема: {theme}")

    def show_connection_dialog(self):
        dialog = ServerConnectionDialog(self, self.server_url, self.current_theme)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.server_url = dialog.get_server_url()
            self.settings.setValue("server_url", self.server_url)
            if self.api_client:
                self.api_client = self.api_client.__class__(self.server_url)
            new_theme = dialog.get_theme()
            if new_theme != self.current_theme:
                self.apply_theme(new_theme)
            self.check_server_connection()

    def check_server_connection(self):
        if self.api_client and self.api_client.check_connection():
            self.statusBar().showMessage(f"Подключено к {self.server_url}")
            self.monitoring_panel.add_log_message("Подключение к серверу успешно")
        else:
            self.statusBar().showMessage("Сервер недоступен (или демо-режим)")
            self.monitoring_panel.add_log_message("Не удалось подключиться к серверу")
    #endregion

    #region File & Optimization Control
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите файл с фигурами", "", "DXF Files (*.dxf);;STEP Files (*.step *.stp);;All Files (*)")
        if filename:
            self.current_file = filename
            self.file_label.setText(Path(filename).name)
            self.monitoring_panel.add_log_message(f"Выбран файл: {filename}")
            if self.api_client:
                try:
                    shapes_info = self.api_client.get_shapes_info(filename)
                    num_shapes = shapes_info.get('num_shapes', 0)
                    self.shapes_info_label.setText(f"{num_shapes} фигур")
                    self.control_panel.start_button.setEnabled(True)
                except Exception as e:
                    self.shapes_info_label.setText("Ошибка")
                    self.monitoring_panel.add_log_message(f"Ошибка получения информации: {e}")
                    self.control_panel.start_button.setEnabled(False)
            else:
                self.shapes_info_label.setText("Демо: файл выбран")
                self.control_panel.start_button.setEnabled(True)

    def start_optimization(self):
        if not self.current_file:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл для оптимизации")
            return
        
        self.control_panel.enable_controls(False)
        self.control_panel.enable_export(False)
        
        if self.api_client:
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
                    use_original_positions=True,
                    history_sample_rate=5,
                    progress_callback_interval=20 
                )
                self.current_task_id = task_id
                self.monitoring_panel.add_log_message(f"Задача запущена: {task_id}")
                self.task_info.setText(f"ID задачи: {task_id}\nСтатус: В ожидании...")
                
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
        else:
            self.monitoring_panel.add_log_message("Демо-режим: Оптимизация запущена (без сервера)")
            self.control_panel.enable_controls(False)

    def stop_optimization(self):
        if self.optimization_thread:
            self.optimization_thread.stop()
            self.monitoring_panel.add_log_message("Оптимизация остановлена пользователем")
        self.control_panel.enable_controls(True)
    #endregion

    #region Optimization Callbacks
    def on_status_update(self, status: dict):
        """Обновление интерфейса при получении статуса от сервера (Реализация Предложения 3)"""
        
        current_shapes = status.get('current_shapes') or []
        print(f"[DEBUG] Status update: {status.get('status')}")
        print(f"[DEBUG] current_shapes: {len(current_shapes)} shapes")
        print(f"[DEBUG] history: {status.get('history', {})}")
        
        utilization = status.get('utilization', 0.0) or 0.0
        
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
        
        # --- Обновление таблицы агентов (только из current_shapes) ---
        current_shapes = status.get('current_shapes') or []
        # print(f"Проверка current_shapes:{current_shapes}")
        self.visualization.update_from_status(status)
        
        if current_shapes:
            print("Вызов update_agents_table")
            self.monitoring_panel.update_agents_table(current_shapes)

        # --- ИСПРАВЛЕНИЕ П.2 и П.3: Обновление графиков по событию ---
        # Ожидаем, что сервер в статусе возвращает историю: {'history': {'iterations': [...], 'energies': [...], 'utilizations': [...]}}
        history = status.get('history', {})
        if history:
            self.monitoring_panel.update_energy_graphs(
                history.get('iterations', []),
                history.get('energies', []),
                history.get('utilizations', [])
            )
        
        
        

    def on_optimization_finished(self, status: dict):
        self.monitoring_panel.add_log_message(f"Оптимизация завершена: {status.get('message', '')}")
        self.control_panel.enable_export(True)
        self.control_panel.stop_button.setEnabled(False)
        utilization = status.get('utilization') if status else 0
        self.statusBar().showMessage(f"Оптимизация завершена. Использование: {utilization:.1f}%")
        try:
            if self.api_client:
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
    #endregion

    #region Export & Visualization
    def download_result(self, format_type: str):
        if not self.current_task_id: return
        output_dir = QFileDialog.getExistingDirectory(self, "Выберите директорию для сохранения")
        if not output_dir: return
        try:
            output_path = Path(output_dir) / f"iagi_result.{format_type}"
            if self.api_client:
                self.api_client.download_result(self.current_task_id, format_type, str(output_path))
            self.monitoring_panel.add_log_message(f"Результат сохранен: {output_path}")
            QMessageBox.information(self, "Экспорт завершен", f"Файл сохранен:\n{output_path}")
        except Exception as e:
            self.monitoring_panel.add_log_message(f"Ошибка экспорта: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить результат: {e}")

    def refresh_visualization(self):
        if not self.current_task_id: return
        try:
            if self.api_client:
                status = self.api_client.get_task_status(self.current_task_id)
                utilization = status.get('utilization', 0.0) or 0.0
                if 'result' in status and isinstance(status['result'], list):
                    self.visualization.update_visualization(status['result'], utilization)
                else:
                    self.visualization.update_visualization([], utilization)
        except Exception as e:
            self.monitoring_panel.add_log_message(f"Ошибка обновления визуализации: {e}")

    def export_visualization_png(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить визуализацию", "", "PNG Files (*.png)")
        if filename:
            self.visualization.export_to_png(filename)
            self.monitoring_panel.add_log_message(f"Визуализация экспортирована: {filename}")

    def export_visualization_svg(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить визуализацию", "", "SVG Files (*.svg)")
        if filename:
            self.visualization.export_to_svg(filename)
            self.monitoring_panel.add_log_message(f"Визуализация экспортирована: {filename}")
    #endregion

    def log_message(self, message: str):
        """Вспомогательный метод для логирования"""
        self.monitoring_panel.add_log_message(message)
    
    def show_about(self):
        QMessageBox.about(self, "О программе", "ИАГИ Клиент\n\
            \nКлиентское приложение для системы оптимизации раскроя плоских деталей с использованием гравитационной имитации.\n\
                \nВерсия 2.0\
                    \nПодключается к серверу ИАГИ через HTTP API\
                    \nВключает визуализацию результатов и мониторинг процесса")