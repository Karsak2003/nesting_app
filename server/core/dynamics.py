import numpy as np
from typing import Tuple, Any, Optional

from scipy.spatial import distance

C_interaction_radius:float = 100.0 # по умолчанию 25


class GravitationalDynamics:
    """
    Физическая модель на основе гравитационной имитации
    """
    
    def __init__(
        self, 
        gravity_strength: float = 9.8, 
        damping_linear: float = 0.5, 
        damping_angular: float = 0.3
    ):
        """
        Инициализация физической модели
        :param gravity_strength: сила гравитации
        :param damping_linear: коэффициент линейного демпфирования
        :param damping_angular: коэффициент углового демпфирования
        """
        self.gravity_strength = gravity_strength
        self.damping_linear = damping_linear
        self.damping_angular = damping_angular
        self.regularization_epsilon: float = 0.1  # мм
        
    def calculate_repulsive_force(
        self, 
        shape1: Any, 
        shape2: Any, 
        collision_detector: Any, 
        min_gap: float = 0.0, 
        repulsion_strength: float = 200.0, 
        exponent: float = 2.0,
        agent1: Optional[Any] = None, 
        agent2: Optional[Any] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Расчет силы отталкивания между двумя фигурами
        """
        distance, cp1, cp2, normal = collision_detector.calculate_min_distance(
            shape1, shape2, min_gap
        )
         
        interaction_radius = max(100.0, min_gap * 3.0)
        if distance > interaction_radius:
            return np.array([0.0, 0.0]), 0.0

        if distance < min_gap:
            penetration = min_gap - distance
            # Усиленная сила при проникновении: квадратичная зависимость
            force_magnitude = repulsion_strength * (penetration ** 2) * 5.0
            # Увеличиваем максимальную силу для разделения глубоко проникших фигур
            max_force = 5000.0  # было 1000.0
        else:
            effective_distance = max(distance, self.regularization_epsilon)
            base_force = repulsion_strength / (effective_distance ** 2)
            if distance > 20.0:
                distance_factor = 1.0 / (1.0 + (distance - 20.0) / 30.0)
                base_force *= distance_factor
            force_magnitude = base_force
        
        if agent1 is not None and agent2 is not None:
            relative_velocity = agent1.velocity - agent2.velocity
            velocity_component = np.dot(relative_velocity, normal)
            if velocity_component < 0:
                velocity_factor = 1.0 + abs(velocity_component) * 0.1
                force_magnitude *= velocity_factor
        
        # Ограничение максимальной силы для численной устойчивости
        max_force = 5000.0 if distance < 0 else 1000.0  # Усиленная сила при проникновении
        force_magnitude = min(force_magnitude, max_force)
        
        # Направление силы отталкивания
        force_direction = normal if distance < min_gap else -normal
        force_vector = force_magnitude * force_direction
        
        # Расчет вращающего момента
        lever_arm1 = cp1 - shape1.centroid
        torque1 = np.cross(lever_arm1, force_vector)
        
        return force_vector, torque1
    
    def calculate_boundary_forces(
        self, 
        shape: Any, 
        sheet_size: Tuple[float, float], 
        min_gap: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Расчет сил отталкивания от границ листа
        """
        forces = np.array([0.0, 0.0])
        torque = 0.0
        
        boundary_stiffness = 100.0
        boundary_attraction_zone = 50.0
        boundary_attraction = 0.5
        
        # Расстояния до границ листа
        min_x, min_y, max_x, max_y = shape.polygon.bounds
        width, height = sheet_size
        
        # Левая граница
        if min_x < min_gap:
            penetration = min_gap - min_x
            force_magnitude = boundary_stiffness * penetration ** 2
            forces[0] += force_magnitude
        elif min_x < min_gap + boundary_attraction_zone:
            distance_to_left = min_x - min_gap
            force_magnitude = boundary_attraction * distance_to_left * shape.area * 0.08  # Увеличено с 0.03
            forces[0] -= force_magnitude  # Отрицательное направление = влево
        else:
            distance_to_left = min_x - min_gap
            if distance_to_left < sheet_size[0] * 0.5:  # Только для левой половины листа
                weak_attraction = 0.1 * (1.0 - distance_to_left / (sheet_size[0] * 0.5)) * shape.area * 0.02
                forces[0] -= weak_attraction
        
        # Правая граница
        if max_x > width - min_gap:
            penetration = max_x - (width - min_gap)
            force_magnitude = boundary_stiffness * penetration ** 2
            forces[0] -= force_magnitude
        elif max_x > width - min_gap - boundary_attraction_zone:
            distance_to_right = (width - min_gap) - max_x
            force_magnitude = boundary_attraction * distance_to_right * shape.area * 0.03
            forces[0] += force_magnitude
        
        # Нижняя граница
        if min_y < min_gap:
            penetration = min_gap - min_y
            force_magnitude = 500.0 * penetration ** 2
            forces[1] += force_magnitude
        elif min_y > min_gap:
            distance_to_bottom = min_y - min_gap
            base_gravity = self.gravity_strength * shape.area * 0.04  # Увеличено с 0.05
            distance_factor = (distance_to_bottom / 100.0) ** 1.5
            additional_gravity = distance_factor * shape.area * 0.25  # Увеличено с 0.15
            force_magnitude = base_gravity + additional_gravity
            forces[1] -= force_magnitude
        
        if max_y > height - min_gap:
            penetration = max_y - (height - min_gap)
            force_magnitude = 500.0 * penetration ** 2
            forces[1] -= force_magnitude
            
        return forces, torque
    
    def verlet_integration(
        self, 
        agent: Any, 
        force: np.ndarray, 
        torque: float, 
        dt: float = 0.01
    ) -> Any:
        """
        Численное интегрирование методом Верле с демпфированием
        """
        if not hasattr(agent, '_prev_position'):
            agent._prev_position = agent.position.copy()
        if not hasattr(agent, '_prev_angle'):
            agent._prev_angle = agent.angle
            
        prev_position = agent._prev_position.copy()
        prev_angle = agent._prev_angle
        
        acceleration = force / (agent.mass + 1e-6)  # Избежание деления на ноль
        angular_acceleration = torque / (agent.moment_of_inertia + 1e-6)
        
        current_position = agent.position.copy()
        current_angle = agent.angle
        
        new_position = 2.0 * agent.position - prev_position + acceleration * dt * dt
        new_angle = 2.0 * agent.angle - prev_angle + angular_acceleration * dt * dt
        
        velocity = (new_position - prev_position) / (2.0 * dt)
        angular_velocity = (new_angle - prev_angle) / (2.0 * dt)
        
        velocity_norm = np.linalg.norm(velocity)
        stability_threshold = 1.0
        
        if velocity_norm < stability_threshold:
            damping_factor_linear = 1.0 - (self.damping_linear * 0.5)
            damping_factor_angular = 1.0 - (self.damping_angular * 0.5)
        else:
            damping_factor_linear = 1.0 - self.damping_linear
            damping_factor_angular = 1.0 - self.damping_angular
        
        velocity *= damping_factor_linear
        angular_velocity *= damping_factor_angular
        
        agent.position = prev_position + velocity * 2.0 * dt
        agent.angle = prev_angle + angular_velocity * 2.0 * dt
        
        agent.velocity = velocity
        agent.angular_velocity = angular_velocity
        
        agent._prev_position = current_position
        agent._prev_angle = current_angle
        
        # Применение технологических ограничений на ориентацию
        if agent.orientation_constraints:
            allowed_angles = agent.orientation_constraints
            # Найти ближайший допустимый угол
            diffs = [(angle - agent.angle) % 360 for angle in allowed_angles]
            min_diff_idx = np.argmin(np.abs(diffs))
            agent.angle = allowed_angles[min_diff_idx]
            agent._prev_angle = agent.angle
        
        if hasattr(agent, 'sheet_size') and agent.sheet_size is not None:
            width, height = agent.sheet_size
            min_gap = getattr(agent, 'min_gap', 0.0)
            
            transformed_shape = agent.get_transformed_shape()
            bbox = transformed_shape.get_bounding_box()
            min_x, min_y, max_x, max_y = bbox
            
            correction = np.array([0.0, 0.0])
            
            if min_x < min_gap:
                correction[0] = min_gap - min_x
            if max_x > width - min_gap:
                correction[0] = (width - min_gap) - max_x
            if min_y < min_gap:
                correction[1] = min_gap - min_y
            if max_y > height - min_gap:
                correction[1] = (height - min_gap) - max_y
            
            if np.any(correction != 0):
                agent.position += correction
                agent._prev_position = agent.position.copy()
            
        return agent