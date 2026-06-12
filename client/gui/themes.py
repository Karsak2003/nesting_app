"""
Цветовые палитры для клиентского приложения.
Поддерживает три темы: светлая, тёмная и системная.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ColorPalette:
    """Базовый класс цветовой палитры."""
    
    # Основные цвета
    window_bg: str = "#FFFFFF"
    window_text: str = "#000000"
    
    # Цвета виджетов
    button_bg: str = "#E1E1E1"
    button_text: str = "#000000"
    button_hover: str = "#D0D0D0"
    button_pressed: str = "#C0C0C0"
    
    # Поля ввода
    input_bg: str = "#FFFFFF"
    input_text: str = "#000000"
    input_border: str = "#CCCCCC"
    input_focus_border: str = "#3A7BD5"
    
    # Списки и таблицы
    list_bg: str = "#FFFFFF"
    list_text: str = "#000000"
    list_selected_bg: str = "#3A7BD5"
    list_selected_text: str = "#FFFFFF"
    list_alternate_bg: str = "#F5F5F5"
    
    # Группы и рамки
    group_bg: str = "#F9F9F9"
    group_title: str = "#333333"
    border_color: str = "#CCCCCC"
    
    # Статус бар
    statusbar_bg: str = "#F0F0F0"
    statusbar_text: str = "#333333"
    
    # Прогресс бар
    progress_bg: str = "#E0E0E0"
    progress_bar: str = "#3A7BD5"
    
    # Сообщения
    success_color: str = "#28A745"
    warning_color: str = "#FFC107"
    error_color: str = "#DC3545"
    info_color: str = "#17A2B8"
    
    # Акценты
    primary_color: str = "#3A7BD5"
    secondary_color: str = "#6C757D"
    accent_color: str = "#007BFF"
    
    # Градиенты (для кнопок и заголовков)
    gradient_start: str = "#4A90E2"
    gradient_end: str = "#3A7BD5"
    
    # Прозрачности (для hover эффектов)
    hover_overlay: str = "rgba(0, 0, 0, 0.05)"
    selected_overlay: str = "rgba(58, 123, 213, 0.1)"


@dataclass
class DarkPalette(ColorPalette):
    """Тёмная цветовая палитра."""
    
    # Основные цвета
    window_bg: str = "#2D2D2D"
    window_text: str = "#FFFFFF"
    
    # Цвета виджетов
    button_bg: str = "#3D3D3D"
    button_text: str = "#FFFFFF"
    button_hover: str = "#4D4D4D"
    button_pressed: str = "#5D5D5D"
    
    # Поля ввода
    input_bg: str = "#3D3D3D"
    input_text: str = "#FFFFFF"
    input_border: str = "#555555"
    input_focus_border: str = "#5A9FD5"
    
    # Списки и таблицы
    list_bg: str = "#2D2D2D"
    list_text: str = "#FFFFFF"
    list_selected_bg: str = "#5A9FD5"
    list_selected_text: str = "#FFFFFF"
    list_alternate_bg: str = "#353535"
    
    # Группы и рамки
    group_bg: str = "#353535"
    group_title: str = "#E0E0E0"
    border_color: str = "#555555"
    
    # Статус бар
    statusbar_bg: str = "#252525"
    statusbar_text: str = "#E0E0E0"
    
    # Прогресс бар
    progress_bg: str = "#3D3D3D"
    progress_bar: str = "#5A9FD5"
    
    # Сообщения
    success_color: str = "#4CAF50"
    warning_color: str = "#FFC107"
    error_color: str = "#F44336"
    info_color: str = "#2196F3"
    
    # Акценты
    primary_color: str = "#5A9FD5"
    secondary_color: str = "#9E9E9E"
    accent_color: str = "#64B5F6"
    
    # Градиенты
    gradient_start: str = "#64B5F6"
    gradient_end: str = "#5A9FD5"
    
    # Прозрачности
    hover_overlay: str = "rgba(255, 255, 255, 0.05)"
    selected_overlay: str = "rgba(90, 159, 213, 0.15)"


@dataclass
class LightPalette(ColorPalette):
    """Светлая цветовая палитра (по умолчанию)."""
    # Использует значения базового класса
    pass


def get_system_theme() -> str:
    """
    Определяет системную тему.
    Возвращает 'dark' или 'light'.
    """
    import sys
    
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize'
            ) as key:
                value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
                return 'light' if value == 1 else 'dark'
        except Exception:
            return 'light'
    
    elif sys.platform == 'darwin':
        try:
            import subprocess
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                capture_output=True,
                text=True
            )
            return 'dark' if result.returncode == 0 else 'light'
        except Exception:
            return 'light'
    
    else:
        # Linux и другие платформы
        # Проверяем переменные окружения
        import os
        desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
        
        if 'gnome' in desktop_env or 'unity' in desktop_env:
            try:
                import subprocess
                result = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.desktop.interface', 'color-scheme'],
                    capture_output=True,
                    text=True
                )
                if 'dark' in result.stdout.lower():
                    return 'dark'
            except Exception:
                pass
        
        return 'light'


def get_palette(theme: str = 'system') -> ColorPalette:
    """
    Получает цветовую палитру для указанной темы.
    
    Args:
        theme: 'light', 'dark', или 'system'
    
    Returns:
        Экземпляр ColorPalette
    """
    if theme == 'system':
        system_theme = get_system_theme()
        theme = system_theme
    
    if theme == 'dark':
        return DarkPalette()
    else:
        return LightPalette()


def generate_stylesheet(palette: ColorPalette) -> str:
    """
    Генерирует QSS stylesheet для применения палитры.
    
    Args:
        palette: Экземпляр ColorPalette
    
    Returns:
        Строка с QSS стилями
    """
    stylesheet = f"""
    /* Основные стили окна */
    QMainWindow, QDialog {{
        background-color: {palette.window_bg};
        color: {palette.window_text};
    }}
    
    /* Кнопки */
    QPushButton {{
        background-color: {palette.button_bg};
        color: {palette.button_text};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        padding: 6px 12px;
        min-width: 80px;
    }}
    
    QPushButton:hover {{
        background-color: {palette.button_hover};
    }}
    
    QPushButton:pressed {{
        background-color: {palette.button_pressed};
    }}
    
    QPushButton:disabled {{
        background-color: {palette.button_bg};
        color: {palette.secondary_color};
        opacity: 0.6;
    }}
    
    /* Поля ввода */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {palette.input_bg};
        color: {palette.input_text};
        border: 1px solid {palette.input_border};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {palette.input_focus_border};
    }}
    
    /* Выпадающие списки */
    QComboBox {{
        background-color: {palette.input_bg};
        color: {palette.input_text};
        border: 1px solid {palette.input_border};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QComboBox:hover {{
        border: 1px solid {palette.input_focus_border};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {palette.list_bg};
        color: {palette.list_text};
        border: 1px solid {palette.input_border};
        selection-background-color: {palette.list_selected_bg};
        selection-color: {palette.list_selected_text};
    }}
    
    /* Списки */
    QListWidget, QTableWidget, QTreeWidget {{
        background-color: {palette.list_bg};
        color: {palette.list_text};
        border: 1px solid {palette.input_border};
        border-radius: 4px;
    }}
    
    QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {palette.list_selected_bg};
        color: {palette.list_selected_text};
    }}
    
    QListWidget::item:alternate, QTableWidget::item:alternate {{
        background-color: {palette.list_alternate_bg};
    }}
    
    QListWidget::item:hover, QTableWidget::item:hover, QTreeWidget::item:hover {{
        background-color: {palette.hover_overlay};
    }}
    
    /* Заголовки таблиц */
    QHeaderView::section {{
        background-color: {palette.group_bg};
        color: {palette.group_title};
        border: 1px solid {palette.border_color};
        padding: 4px 8px;
        font-weight: bold;
    }}
    
    /* Группы */
    QGroupBox {{
        background-color: {palette.group_bg};
        color: {palette.group_title};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 12px;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 8px;
    }}
    
    /* Вкладки */
    QTabWidget::pane {{
        border: 1px solid {palette.border_color};
        border-radius: 4px;
    }}
    
    QTabBar::tab {{
        background-color: {palette.group_bg};
        color: {palette.group_title};
        border: 1px solid {palette.border_color};
        border-bottom: none;
        padding: 6px 12px;
        margin-right: 2px;
    }}
    
    QTabBar::tab:selected {{
        background-color: {palette.window_bg};
        border-top: 2px solid {palette.primary_color};
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {palette.button_hover};
    }}
    
    /* Прогресс бар */
    QProgressBar {{
        background-color: {palette.progress_bg};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        text-align: center;
        color: {palette.window_text};
    }}
    
    QProgressBar::chunk {{
        background-color: {palette.progress_bar};
        border-radius: 3px;
    }}
    
    /* Статус бар */
    QStatusBar {{
        background-color: {palette.statusbar_bg};
        color: {palette.statusbar_text};
        border-top: 1px solid {palette.border_color};
    }}
    
    /* Метки */
    QLabel {{
        color: {palette.window_text};
    }}
    
    QLabel[cssClass="success"] {{
        color: {palette.success_color};
        font-weight: bold;
    }}
    
    QLabel[cssClass="warning"] {{
        color: {palette.warning_color};
        font-weight: bold;
    }}
    
    QLabel[cssClass="error"] {{
        color: {palette.error_color};
        font-weight: bold;
    }}
    
    QLabel[cssClass="info"] {{
        color: {palette.info_color};
        font-weight: bold;
    }}
    
    /* Чекбоксы и радиокнопки */
    QCheckBox, QRadioButton {{
        color: {palette.window_text};
        spacing: 8px;
    }}
    
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
    }}
    
    /* Слайдеры */
    QSlider::groove:horizontal {{
        border: 1px solid {palette.border_color};
        height: 8px;
        background: {palette.progress_bg};
        border-radius: 4px;
    }}
    
    QSlider::handle:horizontal {{
        background: {palette.primary_color};
        border: 1px solid {palette.border_color};
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    
    QSlider::sub-page:horizontal {{
        background: {palette.progress_bar};
        border-radius: 4px;
    }}
    
    /* Scroll bars */
    QScrollBar:vertical {{
        background: {palette.progress_bg};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {palette.secondary_color};
        min-height: 20px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {palette.primary_color};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar:horizontal {{
        background: {palette.progress_bg};
        height: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal {{
        background: {palette.secondary_color};
        min-width: 20px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:horizontal:hover {{
        background: {palette.primary_color};
    }}
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    
    /* Меню */
    QMenuBar {{
        background-color: {palette.group_bg};
        color: {palette.group_title};
        border-bottom: 1px solid {palette.border_color};
    }}
    
    QMenuBar::item:selected {{
        background-color: {palette.list_selected_bg};
        color: {palette.list_selected_text};
    }}
    
    QMenu {{
        background-color: {palette.window_bg};
        color: {palette.window_text};
        border: 1px solid {palette.border_color};
    }}
    
    QMenu::item:selected {{
        background-color: {palette.list_selected_bg};
        color: {palette.list_selected_text};
    }}
    
    QMenu::separator {{
        height: 1px;
        background: {palette.border_color};
        margin: 4px 8px;
    }}
    
    /* Tooltips */
    QToolTip {{
        background-color: {palette.window_bg};
        color: {palette.window_text};
        border: 1px solid {palette.border_color};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    /* SpinBox */
    QSpinBox, QDoubleSpinBox {{
        background-color: {palette.input_bg};
        color: {palette.input_text};
        border: 1px solid {palette.input_border};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {palette.input_focus_border};
    }}
    
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        width: 16px;
        border: none;
        background: {palette.button_bg};
    }}
    
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {palette.button_hover};
    }}
    """
    
    return stylesheet
