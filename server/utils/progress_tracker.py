"""
Потокобезопасный трекер прогресса оптимизации
Собирает историю (итерации, энергии, утилизации) и снимки текущего состояния фигур
для real-time визуализации на клиенте.
"""
import threading
import logging
from typing import List, Dict, Any, Optional
import math 

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Вызываемый объект (callable), передаваемый в алгоритмы как progress_callback.
    Автоматически:
      - прореживает запись в историю (каждые callback_interval итераций);
      - строит снимок состояния фигур (каждые snapshot_interval итераций);
      - обновляет основной статус задачи в optimization_tasks.
    
    Все операции потокобезопасны благодаря threading.Lock.
    """
    
    def __init__(
        self,
        task_id: str,
        task_dict: Dict[str, Any],
        lock: threading.Lock,
        snapshot_interval: int = 5,
        callback_interval: int = 20
    ):
        """
        :param task_id: Идентификатор задачи
        :param task_dict: Ссылка на optimization_tasks[task_id]
        :param lock: Блокировка для потокобезопасности
        :param snapshot_interval: Частота построения снимка current_shapes (в итерациях)
        :param callback_interval: Частота записи в history (в итерациях)
        """
        self.task_id = task_id
        self.task_dict = task_dict
        self.history = task_dict['history']
        self.lock = lock
        self.snapshot_interval = max(1, snapshot_interval)
        self.callback_interval = max(1, callback_interval)
        self.iteration = 0
    
    def __call__(
        self,
        progress: int,
        agents: list,
        utilization: float,
        energy: float
    ) -> None:
        """
        Сигнатура соответствует вызовам в sequential.py и parallel.py:
        progress_callback(progress, agents, utilization, energy)
        """
        with self.lock:
            self.iteration += 1
            is_final = (progress >= 100)
            
            # === 1. Прореживание записи в историю ===
            if self.iteration % self.callback_interval == 0 or is_final:
                self.history['iterations'].append(self.iteration)
                self.history['energies'].append(round(energy, 4))
                self.history['utilizations'].append(round(utilization, 4))
                
                # Обновление основного статуса задачи
                self.task_dict['progress'] = min(100, progress)
                self.task_dict['utilization'] = utilization
                self.task_dict['message'] = (
                    f'Итерация {self.iteration}, '
                    f'E={energy:.2f}, U={utilization:.2f}%'
                )
            
            # === 2. Построение снимка текущего состояния фигур ===
            if self.iteration % self.snapshot_interval == 0 or is_final:
                if agents is not None:
                    try:
                        self.task_dict['current_shapes'] = self._build_shapes_snapshot(agents)
                    except Exception as e:
                        logger.warning(f"[ProgressTracker] Ошибка построения снимка: {e}")
                            
    def _build_shapes_snapshot(self, agents: list) -> List[Dict[str, Any]]:
        """
        Формирует сериализуемый снимок состояния всех агентов.
        Контур — List[List[float, float]] (не Shapely-объект!).
        Все float-значения округлены до 3-4 знаков для уменьшения payload.
        """
        snapshot = []
        for idx, agent in enumerate(agents):
            try:
                transformed = agent.get_transformed_shape()
                
                # 1. Контур — список [[x, y], ...] (НЕ Shapely-объект!)
                if hasattr(transformed, 'exterior'):
                    # Shapely-объект напрямую
                    contour = [[round(float(x), 3), round(float(y), 3)] 
                            for x, y in transformed.exterior.coords[:-1]]
                elif hasattr(transformed, 'polygon'):
                    # PolygonShape с атрибутом polygon
                    contour = [[round(float(x), 3), round(float(y), 3)] 
                            for x, y in transformed.polygon.exterior.coords[:-1]]
                else:
                    contour = []
                
                # 2. Центроид (центр масс)
                centroid = [
                    round(float(agent.position[0]), 3),
                    round(float(agent.position[1]), 3)
                ]
                
                # 3. Скорости (с защитой от отсутствия атрибутов)
                is_frozen = bool(getattr(agent, 'is_frozen', False))
                
                if is_frozen:
                    # Для замороженных агентов скорости равны нулю
                    velocity = [0.0, 0.0]
                    angular_velocity = 0.0
                else:
                    velocity_raw = getattr(agent, 'velocity', [0.0, 0.0])
                    if hasattr(velocity_raw, 'tolist'):
                        velocity_raw = velocity_raw.tolist()
                    velocity = [round(float(v), 4) for v in velocity_raw]
                    
                    angular_velocity_raw = float(getattr(agent, 'angular_velocity', 0.0))
                    angular_velocity = round(angular_velocity_raw, 4)
                
                # 4. Защита от NaN/Inf
                def safe_float(val: float, default: float = 0.0) -> float:
                    return val if math.isfinite(val) else default
                
                contour = [[safe_float(x), safe_float(y)] for x, y in contour]
                centroid = [safe_float(c) for c in centroid]
                velocity = [safe_float(v) for v in velocity]
                angular_velocity = safe_float(angular_velocity)
                
                snapshot.append({
                    'id': idx,
                    'name': getattr(agent.shape, 'name', f'part_{idx+1}'),
                    'contour': contour,
                    'centroid': centroid,
                    'angle': round(float(getattr(agent, 'angle', 0.0)), 3),
                    'priority': int(getattr(agent, 'priority', 1)),
                    'is_frozen': is_frozen,
                    'velocity': velocity,
                    'angular_velocity': angular_velocity
                })
            except Exception as e:
                logger.warning(f"[ProgressTracker] Не удалось снять снапшот агента {idx}: {e}")
                continue
        
        return snapshot
