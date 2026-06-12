import threading
from typing import List, Dict, Any

class ProgressTracker:
    def __init__(self, task_id: str, history_dict: Dict[str, List], lock: threading.Lock):
        self.task_id = task_id
        self.history = history_dict
        self.lock = lock
        self.iteration = 0

    def __call__(self, progress: int, agents: list, utilization: float, energy: float):
        """
        Сигнатура точно соответствует ожидаемой в sequential.py и parallel.py:
        Callable[[int, list, float, float], None]
        """
        with self.lock:
            self.iteration += 1
            
            # Прореживание данных: сохраняем каждую 5-ю итерацию или финал (progress >= 99)
            # Это предотвращает раздувание памяти и перегрузку сети при отдаче JSON
            if self.iteration % 5 == 0 or progress >= 99:
                self.history['iterations'].append(self.iteration)
                self.history['energies'].append(round(energy, 4))
                self.history['utilizations'].append(round(utilization, 4))
            
            # Обновление основного статуса задачи (предполагается, что ссылка на optimization_tasks передаётся или обновляется через замыкание/глобальный доступ, 
            # либо трекер возвращает dict для обновления в main.py)