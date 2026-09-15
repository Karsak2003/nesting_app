"""
HTTP-клиент и фоновый поток для взаимодействия с сервером ИАГИ.
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from PySide6.QtCore import QThread, Signal as pyqtSignal


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
        use_original_positions: bool = True,
        history_sample_rate: int = 5,              # ← НОВОЕ
        progress_callback_interval: int = 20       # ← НОВОЕ
    ) -> str:
        """    
        Запуск оптимизации и возврат task_id.
    
        Args:
            history_sample_rate: сохранять снимок каждые N итераций (прореживание истории)
            progress_callback_interval: частота вызова progress_callback (в итерациях)"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f)}
                history_sample_rate = max(1, int(history_sample_rate))
                progress_callback_interval = max(1, int(progress_callback_interval))

                data = {
                    'profile': profile,
                    'algorithm': algorithm,
                    'technology': technology,
                    'min_gap': min_gap,
                    'time_limit': time_limit,
                    'sheet_width': sheet_width,
                    'sheet_height': sheet_height,
                    'use_original_positions': use_original_positions,
                    'history_sample_rate': history_sample_rate,              # ← НОВОЕ
                    'progress_callback_interval': progress_callback_interval # ← НОВОЕ
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
        
    def get_task_status(self, task_id: str, since_iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Получение статуса задачи.
        
        Args:
            task_id: Идентификатор задачи
            since_iteration: Если указан, вернуть только историю после этой итерации
        
        Returns:
            Словарь с статусом задачи
        """
        try:
            params = {}
            if since_iteration is not None:
                params['since_iteration'] = since_iteration
            
            response = self.session.get(
                f"{self.base_url}/api/status/{task_id}",
                params=params,  # ← Передаём query-параметр
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
        """Мониторинг статуса задачи с гибридным подходом"""
        try:
            last_iteration = 0  # ← Локальное хранение
            
            while self.running:
                # Запрашиваем дельту истории
                status = self.api_client.get_task_status(
                    self.task_id,
                    since_iteration=last_iteration
                )
                
                # Обновляем last_iteration на основе полученной истории
                history = status.get('history', {})
                iterations = history.get('iterations', [])
                if iterations:
                    last_iteration = max(iterations)
                
                # Передаём статус в GUI
                self.status_update.emit(status)
                
                if status['status'] in ['completed', 'failed']:
                    self.finished_signal.emit(status)
                    break
                
                time.sleep(self.poll_interval)
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        self.running = False