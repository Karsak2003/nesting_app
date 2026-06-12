"""
Пакет графического интерфейса клиента ИАГИ
"""
from .config import get_client_config, save_client_config
from .visualization import RealTimeVisualization
from .dialogs import ServerConnectionDialog, LoadFileDialog, SplashScreen
from .panels import ControlPanel, MonitoringPanel
from .main_window import MainWindow

__all__ = [
    "get_client_config", "save_client_config",
    "RealTimeVisualization",
    "ServerConnectionDialog", "LoadFileDialog", "SplashScreen",
    "ControlPanel", "MonitoringPanel",
    "MainWindow",
    "APIClient", "OptimizationThread"
]