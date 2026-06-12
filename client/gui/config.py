#region Imports
import json
from pathlib import Path
from typing import Dict, Any
#endregion

def get_client_config() -> Dict[str, Any]:
    """Загрузка конфигурации клиента из существующего JSON файла в корне проекта"""
    # Path(__file__) указывает на gui/config.py
    # .parent.parent поднимает на два уровня вверх, в корень проекта
    config_path = Path(__file__).parent.parent / "config" / "settings.json"
    
    default_config = {
        "server": {"url": "http://localhost:8000", "timeout": 300},
        "theme": "system",
        "local_server": {"enabled": True, "auto_start": False, "port": 8000, "show_console": True},
        "last_used": {"profile": "medium_precision", "algorithm": "sequential", "technology": "laser"}
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Безопасное объединение с дефолтными значениями
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
    """Сохранение конфигурации клиента в существующий JSON файл"""
    config_path = Path(__file__).parent.parent / "config" / "settings.json"
    
    # Гарантируем, что папка config существует (на случай, если её удалят)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)