import numpy as np
from typing import Optional, List, Tuple, Any


class IAGIAgent:
    """
    Интеллектуальный агент гравитационной имитации (ИАГИ)
    Реализует архитектуру BDI (belief-desire-intention)
    """
    def __init__(
        self, 
        shape: Any, 
        position: Optional[Tuple[float, float]] = None, 
        angle: float = 0.0, 
        priority: int = 1
    ):
        """
        Инициализация ИАГИ агента
        :param shape: геометрия фигуры
        :param position: начальная позиция (x, y)
        :param angle: начальный угол поворота
        :param priority: приоритет размещения (чем выше, тем раньше размещается)
        """
        self.shape = shape
        self.original_shape = shape
        
        # Внутреннее состояние (belief)
        self.position = np.array(position) if position is not None else np.array([0.0, 0.0])
        self.angle = float(angle)
        self.velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0
        
        # Физические характеристики
        self.mass = shape.area * 0.1  # Пропорционально площади
        self.moment_of_inertia = shape.moment_of_inertia
        
        # Цели агента (desire)
        self.desired_y = 0.0  # Стремление к нижней границе
        self.minimize_collisions = True
        
        # Параметры агента
        self.priority = priority
        self.is_frozen = False
        
        # Технологические ограничения
        self.orientation_constraints = None  # None означает свободное вращение
        self.min_gap = 0.0
        
        # Параметры для локальной релаксации
        self.perception_radius = 100.0  # мм
        self.stagnation_counter = 0
        self.max_stagnation_steps = 100
        
    def update_beliefs(
        self, 
        neighbors: List[Any], 
        sheet_size: Tuple[float, float], 
        defects: Optional[List[Any]] = None
    ) -> None:
        """
        Обновление знаний агента об окружающей среде (belief)
        """
        self.neighbors = neighbors
        self.sheet_size = sheet_size
        self.defects = defects if defects else []
        
    def compute_intention(self, collision_detector: Any, dynamics: Any) -> Tuple[np.ndarray, float]:
        """
        Формирование намерений агента на основе текущего состояния (intention)
        """
        if self.is_frozen:
            return np.array([0.0, 0.0]), 0.0
        
        total_force = np.array([0.0, 0.0])
        total_torque = 0.0

        self_transformed = self.get_transformed_shape()
        
        # 1. Силы от соседних фигур
        for neighbor in self.neighbors:
            if neighbor is self:
                continue

            neighbor_transformed = neighbor.get_transformed_shape()
                
            force, torque = dynamics.calculate_repulsive_force(
                self_transformed, neighbor_transformed, collision_detector, self.min_gap,
                agent1=self, agent2=neighbor
            )
            total_force += force
            total_torque += torque
        
        # 2. Силы от границ листа
        boundary_forces, boundary_torque = dynamics.calculate_boundary_forces(
            self_transformed, self.sheet_size, self.min_gap
        )
        total_force += boundary_forces
        total_torque += boundary_torque
        
        # 3. Силы от дефектных зон
        for defect in self.defects:
            force, torque = dynamics.calculate_repulsive_force(
                self_transformed, defect, collision_detector, self.min_gap + 2.0
            )
            total_force += force * 2.0  # Усиленное отталкивание от дефектов
            total_torque += torque * 2.0

        return total_force, total_torque
    
    def adaptive_damping(self, velocity_threshold: float = 1.0) -> float:
        """
        Адаптивное демпфирование для стабилизации
        """
        velocity_norm = np.linalg.norm(self.velocity)
        
        if velocity_norm < velocity_threshold:
            # Увеличиваем демпфирование для быстрой стабилизации
            return min(0.95, self.damping_linear * 1.1)
        
        return self.damping_linear
    
    def apply_micro_displacement(self, max_displacement: float = 1.0) -> bool:
        """
        Микросмещение для выхода из локальных минимумов
        """
        if self.stagnation_counter > self.max_stagnation_steps:
            angle = np.random.uniform(0, 2 * np.pi)
            displacement = np.random.uniform(0, max_displacement)
            self.position += np.array([
                displacement * np.cos(angle),
                displacement * np.sin(angle)
            ])
            self.stagnation_counter = 0
            return True
        return False
    
    def get_transformed_shape(self) -> Any:
        """
        Получение геометрии фигуры с учетом текущего положения и поворота
        """
        return self.shape.apply_transformation(self.position, self.angle)