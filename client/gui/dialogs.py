#region Imports
import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                             QLineEdit, QRadioButton, QButtonGroup, QLabel, QComboBox,
                             QDialogButtonBox, QFileDialog, QTextEdit, QDoubleSpinBox,
                             QListWidget, QListWidgetItem, QFrame, QProgressBar, QPushButton)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
#endregion

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
        conn_group = QGroupBox("Параметры подключения")
        conn_layout = QFormLayout()
        
        self.server_edit = QLineEdit()
        self.server_edit.setText(self.current_url)
        conn_layout.addRow("Адрес сервера:", self.server_edit)
        
        preset_layout = QHBoxLayout()
        self.local_radio = QRadioButton("Локальный (localhost:8000)")
        self.local_radio.setChecked(True)
        self.remote_radio = QRadioButton("Удаленный")
        button_group = QButtonGroup(self)
        button_group.addButton(self.local_radio)
        button_group.addButton(self.remote_radio)
        self.local_radio.toggled.connect(lambda checked: self.server_edit.setText("http://localhost:8000" if checked else self.server_edit.text()))
        
        preset_layout.addWidget(self.local_radio)
        preset_layout.addWidget(self.remote_radio)
        conn_layout.addRow("Пресет:", preset_layout)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        theme_group = QGroupBox("Цветовая тема")
        theme_layout = QVBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Системная", "system")
        self.theme_combo.addItem("Светлая", "light")
        self.theme_combo.addItem("Тёмная", "dark")
        for i in range(self.theme_combo.count()):
            if self.theme_combo.itemData(i) == self.current_theme:
                self.theme_combo.setCurrentIndex(i)
                break
        theme_layout.addWidget(QLabel("Выберите тему:"))
        theme_layout.addWidget(self.theme_combo)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_server_url(self) -> str: return self.server_edit.text().rstrip('/')
    def get_theme(self) -> str: return self.theme_combo.currentData()


class LoadFileDialog(QDialog):
    """Диалог для загрузки данных из файлов"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Загрузка данных")
        self.setMinimumWidth(500)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Файл данных:"))
        self.file_edit = QLineEdit()
        file_layout.addWidget(self.file_edit)
        self.browse_button = QPushButton("Обзор...") # Исправлено: добавлен импорт QPushButton выше, если нужен, но он уже в QtWidgets
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)
        
        info_group = QGroupBox("Информация о файле")
        info_layout = QVBoxLayout()
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setFixedHeight(100)
        info_layout.addWidget(self.file_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        import_group = QGroupBox("Настройки импорта")
        import_layout = QFormLayout()
        self.sheet_width = QDoubleSpinBox(); self.sheet_width.setRange(100, 10000); self.sheet_width.setValue(2000)
        import_layout.addRow("Ширина листа (мм):", self.sheet_width)
        self.sheet_height = QDoubleSpinBox(); self.sheet_height.setRange(100, 10000); self.sheet_height.setValue(1000)
        import_layout.addRow("Высота листа (мм):", self.sheet_height)
        self.tolerance_spin = QDoubleSpinBox(); self.tolerance_spin.setRange(0.01, 1.0); self.tolerance_spin.setValue(0.1); self.tolerance_spin.setSingleStep(0.01)
        import_layout.addRow("Точность (мм):", self.tolerance_spin)
        import_group.setLayout(import_layout)
        layout.addWidget(import_group)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "DXF Files (*.dxf);;STEP Files (*.step *.stp);;JSON Files (*.json);;All Files (*)")
        if filename:
            self.file_edit.setText(filename)
            self.update_file_info(filename)

    def update_file_info(self, filename: str):
        if not os.path.exists(filename):
            self.file_info.setText("Файл не найден")
            return
        file_size = os.path.getsize(filename) / 1024
        file_ext = os.path.splitext(filename)[1].lower()
        info = f"Имя файла: {os.path.basename(filename)}\nРазмер: {file_size:.1f} KB\nТип: {file_ext.upper()[1:]}\n"
        if file_ext in ['.dxf', '.step', '.stp']: info += "Формат: CAD-файл с геометрией деталей"
        elif file_ext == '.json': info += "Формат: Конфигурационный файл"
        self.file_info.setText(info)


class SplashScreen(QDialog):
    """Экран-заставка с описанием этапов запуска"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        container = QFrame()
        container.setStyleSheet("QFrame { background-color: #2D2D2D; border-radius: 15px; border: 2px solid #5A9FD5; }")
        container_layout = QVBoxLayout()
        
        title_label = QLabel("ИАГИ Клиент")
        title_label.setStyleSheet("color: #FFFFFF; font-size: 24px; font-weight: bold; padding: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(title_label)
        
        self.steps_list = QListWidget()
        self.steps_list.setStyleSheet("QListWidget { background-color: transparent; color: #E0E0E0; font-size: 14px; border: none; } QListWidget::item { padding: 10px; border-bottom: 1px solid #555555; } QListWidget::item:selected { background-color: #5A9FD5; }")
        stages = ["⏳ Инициализация приложения...", "📂 Загрузка настроек...", "🔌 Подключение к серверу...", "🎨 Применение темы...", "✅ Готово к работе"]
        for stage in stages: self.steps_list.addItem(QListWidgetItem(stage))
        container_layout.addWidget(self.steps_list)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #3D3D3D; border: 1px solid #555555; border-radius: 5px; text-align: center; color: #FFFFFF; } QProgressBar::chunk { background-color: #5A9FD5; border-radius: 4px; }")
        container_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Запуск...")
        self.status_label.setStyleSheet("color: #AAAAAA; font-size: 12px; padding: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.status_label)
        
        container.setLayout(container_layout)
        layout.addWidget(container)
        self.setLayout(layout)

    def update_step(self, step_index: int, progress: int, status_text: str):
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            item.setForeground(QColor("#5A9FD5" if i == step_index else "#E0E0E0"))
            if i == step_index: self.steps_list.setCurrentItem(item)
        self.progress_bar.setValue(progress)
        self.status_label.setText(status_text)
        QApplication.processEvents()