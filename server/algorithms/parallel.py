import numpy as np
import time
import json
from typing import List, Tuple, Dict, Optional
from core.agent import IAGIAgent
from core.collision import CollisionDetector
from core.dynamics import GravitationalDynamics
from core.geometry import PolygonShape
from algorithms.stabilization import apply_stabilization, check_energy_stagnation

def parallel_placement(
    agents: List[IAGIAgent],
    sheet_size: Tuple[float, float],
    collision_detector: CollisionDetector,
    dynamics: GravitationalDynamics,
    min_gap: float,
    defect_zones: Optional[List[PolygonShape]] = None,
    time_limit: float = 300.0,
    energy_threshold: float = 0.1,
    stagnation_steps: int = 50,
    progress_callback=None
) -> List[IAGIAgent]:
    """
    Параллельный алгоритм размещения на основе ИАГИ
    
    :param agents: Список агентов для размещения
    :param sheet_size: Размеры листа (ширина, высота)
    :param collision_detector: Детектор столкновений
    :param dynamics: Физическая модель динамики
    :param min_gap: Минимальный технологический зазор
    :param defect_zones: Список дефектных зон на листе
    :param time_limit: Максимальное время работы алгоритма (секунды)
    :param energy_threshold: Порог сходимости по энергии
    :param stagnation_steps: Количество шагов для обнаружения стагнации
    :param progress_callback: функция обратного вызова (progress, agents, utilization, energy)
    :return: Список агентов с финальными позициями
    """
    start_time = time.time()
    current_time = 0.0
    base_dt = 0.08  # Базовый шаг интегрирования
    
    # Инициализация агентов для параллельного размещения
    num_agents = len(agents)
    if num_agents > 0:
        total_width_used = min(sheet_size[0] * 0.7, num_agents * 150.0)  # Используем 70% ширины или меньше
        start_x = (sheet_size[0] - total_width_used) / 2  # Центрируем
        
        for i, agent in enumerate(agents):
            if num_agents > 1:
                spacing = total_width_used / max(1, num_agents - 1)
                init_x = start_x + (i * spacing)
            else:
                init_x = sheet_size[0] / 2.0

            init_y = sheet_size[1] * 0.3 + np.random.uniform(0, sheet_size[1] * 0.2)
            
            agent.position = np.array([init_x, init_y])
            agent.angle = np.random.uniform(0, 360) if agent.orientation_constraints is None else \
                         np.random.choice(agent.orientation_constraints)
            agent.is_frozen = False
            agent.velocity = np.array([0.0, 0.0])
            agent.angular_velocity = 0.0
    
    # Построение пространственного индекса
    shapes = [agent.shape for agent in agents]
    collision_detector.build_spatial_index(shapes, sheet_size)

    # Инициализация истории энергии для обнаружения стагнации
    energy_history = []
    stagnation_counter = 0
    
    # Основной цикл симуляции
    iteration = 0
    stable = False
    
    print("Запуск параллельного алгоритма размещения...")
    
    while not stable and (time.time() - start_time) < time_limit:
        iteration += 1

        max_velocity = 0.0
        for agent in agents:
            if not agent.is_frozen:
                velocity_norm = np.linalg.norm(agent.velocity)
                max_velocity = max(max_velocity, velocity_norm)

        if max_velocity > 100.0:
            dt = base_dt * 0.5
        elif max_velocity < 1.0:
            dt = base_dt * 1.5
        else:
            dt = base_dt
        
        # 1. Вычисление общей энергии системы для критерия сходимости
        total_energy = calculate_system_energy(agents)
        energy_history.append(total_energy)
        
        # Проверка стагнации энергии
        if check_energy_stagnation(energy_history, stagnation_steps, energy_threshold):
            stagnation_counter += 1
            print(f"Обнаружена стагнация энергии (счетчик: {stagnation_counter}/{stagnation_steps})")
            
            if stagnation_counter >= 3:
                print("Применение механизмов выхода из стагнации...")
                # Случайное смещение части агентов для выхода из локального минимума
                for agent in agents:
                    if not agent.is_frozen and np.random.random() < 0.3:
                        agent.position += np.random.uniform(-5.0, 5.0, size=2)
                stagnation_counter = 0
        else:
            stagnation_counter = 0
        
        # 2. Поиск соседей и обновление состояния для каждого агента
        forces = []
        torques = []
        
        for agent in agents:
            if agent.is_frozen:
                forces.append(np.array([0.0, 0.0]))
                torques.append(0.0)
                continue
            
            # Поиск ближайших соседей с использованием BVH
            nearby_indices = collision_detector.find_nearby_shapes(
                agent.shape, 
                [a.shape for a in agents], 
                agent.perception_radius
            )
            neighbors = [agents[i] for i in nearby_indices if i < len(agents) and agents[i] is not agent]
            
            # Обновление знаний агента об окружающей среде
            agent.update_beliefs(neighbors, sheet_size, defect_zones)
            
            # Расчет сил и моментов
            force, torque = agent.compute_intention(collision_detector, dynamics)
            forces.append(force)
            torques.append(torque)
        
        # 3. Интегрирование уравнений движения для всех агентов
        max_velocity = 0.0
        max_movement = 0.0
        
        for i, agent in enumerate(agents):
            if agent.is_frozen:
                continue
            
            prev_position = agent.position.copy()
            
            # Интегрирование методом Верле
            dynamics.verlet_integration(agent, forces[i], torques[i], dt)
            
            # Отслеживание максимальной скорости и перемещения
            velocity_norm = np.linalg.norm(agent.velocity)
            movement = np.linalg.norm(agent.position - prev_position)
            
            max_velocity = max(max_velocity, velocity_norm)
            max_movement = max(max_movement, movement)
        
        # 4. Проверка коллизий и применение релаксации
        collision_detected = resolve_collisions(agents, collision_detector, min_gap)
        
        # 5. Критерии сходимости
        if not collision_detected and max_velocity < 0.5 and max_movement < 0.1:
            stable = True
        
        # 6. Логирование прогресса и вызов callback
        if iteration % 20 == 0:
            elapsed = time.time() - start_time
            utilization = calculate_material_utilization(agents, sheet_size)
            print(f"Итерация {iteration}: энергия={total_energy:.2f}, "
                  f"max_vel={max_velocity:.2f}, утилизация={utilization:.2f}%, "
                  f"время={elapsed:.1f}/{time_limit:.0f}с")
            
        # Вызов callback для обновления прогресса
        if progress_callback:
            elapsed = time.time() - start_time
            progress = min(int((elapsed / time_limit) * 100), 99)
            utilization = calculate_material_utilization(agents, sheet_size)
            current_energy = calculate_system_energy(agents)
            progress_callback(progress, agents, utilization, current_energy)

        # 7. Экстренная остановка при превышении времени
        if (time.time() - start_time) > time_limit:
            print(f"Достигнут лимит времени ({time_limit}с). Принудительная остановка.")
            break

    # Применение финальной стабилизации
    agents = apply_stabilization(agents, collision_detector, dynamics, min_gap)
    
    elapsed_time = time.time() - start_time
    final_utilization = calculate_material_utilization(agents, sheet_size)
    
    print(f"\nПараллельное размещение завершено за {elapsed_time:.2f} секунд")
    print(f"Финальная утилизация материала: {final_utilization:.2f}%")
    print(f"Количество итераций: {iteration}")
    
    # Финальный вызов callback
    if progress_callback:
        # <-- ДОБАВЛЕНО: Расчет финальной энергии
        final_energy = calculate_system_energy(agents)
        progress_callback(100, agents, final_utilization, final_energy)
    
    return agents

def calculate_system_energy(agents: List[IAGIAgent]) -> float:
    """
    Расчет полной энергии системы согласно формуле (2.64) диссертации:
    E = K + U, где
    K = Σ(0.5·mᵢ·‖vᵢ² + 0.5·Iᵢ·ωᵢ²) — кинетическая (формула 2.50)
    U = Σ(mᵢ·g·y) — гравитационная потенциальная (формула 2.32)
    """
    total_energy = 0.0
    g = 9.8
    
    for agent in agents:
        if getattr(agent, 'is_frozen', False):
            continue
        
        # Поступательная кинетическая энергия
        v = getattr(agent, 'velocity', np.array([0.0, 0.0]))
        v_norm = float(np.linalg.norm(v))
        e_kinetic_trans = 0.5 * agent.mass * (v_norm ** 2)
        
        # Вращательная кинетическая энергия
        # ВАЖНО: момент инерции — у ФИГУРЫ, не у агента!
        omega = float(getattr(agent, 'angular_velocity', 0.0))
        I = float(agent.shape.moment_of_inertia)  # ✅ ИСПРАВЛЕНО
        e_kinetic_rot = 0.5 * I * (omega ** 2)
        
        # Гравитационная потенциальная энергия
        y = float(agent.position[1])
        e_potential = agent.mass * g * y
        
        total_energy += (e_kinetic_trans + e_kinetic_rot + e_potential)
    
    return total_energy

def resolve_collisions(
    agents: List[IAGIAgent],
    collision_detector: CollisionDetector,
    min_gap: float
) -> bool:
    """
    Разрешение коллизий между агентами
    
    :param agents: Список агентов
    :param collision_detector: Детектор столкновений
    :param min_gap: Минимальный технологический зазор
    :return: True если были обнаружены коллизии
    """
    collision_detected = False
    max_iterations = 5
    separation_factor = 1.2
    stability_threshold = 0.5
    for iteration in range(max_iterations):
        iteration_collisions = False

        displacement_accumulator = {i: np.array([0.0, 0.0]) for i in range(len(agents))}
        collision_counts = {i: 0 for i in range(len(agents))}  # Для массового разделения

        collisions = []
        for i, agent1 in enumerate(agents):
            if agent1.is_frozen:
                continue
                
            shape1 = agent1.get_transformed_shape()
            
            for j in range(i + 1, len(agents)):
                agent2 = agents[j]
                if agent2.is_frozen:
                    continue
                    
                shape2 = agent2.get_transformed_shape()
                
                if collision_detector.check_collision(shape1, shape2, min_gap):
                    iteration_collisions = True
                    collision_detected = True
                    
                    relative_velocity = np.linalg.norm(agent1.velocity - agent2.velocity)
                    is_active = relative_velocity > stability_threshold
                    
                    collisions.append((i, j, agent1, agent2, shape1, shape2, is_active))
                    collision_counts[i] += 1
                    collision_counts[j] += 1

        if not iteration_collisions:
            break

        for i, j, agent1, agent2, shape1, shape2, is_active in collisions:
            distance, cp1, cp2, normal = collision_detector.calculate_min_distance(
                shape1, shape2, min_gap
            )

            penetration_depth = -distance if distance < 0 else 0
            
            if penetration_depth > 0:
                collision_count1 = max(collision_counts[i], 1)
                collision_count2 = max(collision_counts[j], 1)
                effective_mass1 = agent1.mass / collision_count1
                effective_mass2 = agent2.mass / collision_count2
                total_effective_mass = effective_mass1 + effective_mass2
                
                if total_effective_mass > 0:
                    ratio1 = effective_mass1 / total_effective_mass
                    ratio2 = effective_mass2 / total_effective_mass
                else:
                    ratio1 = ratio2 = 0.5
                
                if is_active:
                    base_factor = 0.9 + 0.1 * (iteration + 1) / max_iterations
                else:
                    base_factor = 0.7 + 0.1 * (iteration + 1) / max_iterations
                
                displacement_factor = base_factor * separation_factor
                
                displacement = normal * (penetration_depth + min_gap) * displacement_factor
                
                displacement_accumulator[i] += displacement * ratio2
                displacement_accumulator[j] -= displacement * ratio1
        
        for i, displacement in displacement_accumulator.items():
            if np.linalg.norm(displacement) > 1e-6:
                agents[i].position += displacement
        
        if iteration_collisions:
            for i in range(len(agents)):
                if not agents[i].is_frozen and collision_counts[i] > 0:
                    pass
    
    return collision_detected

def calculate_material_utilization(
    agents: List[IAGIAgent],
    sheet_size: Tuple[float, float]
) -> float:
    """
    Расчет коэффициента использования материала

    :param agents: Список агентов
    :param sheet_size: Размеры листа
    :return: Процент использования материала
    """
    total_area = sum(agent.shape.area for agent in agents)
    
    # Расчет фактических границ использованного материала
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for agent in agents:
        transformed_shape = agent.get_transformed_shape()
        bbox = transformed_shape.get_bounding_box()
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])

    if max_x > min_x and max_y > min_y:
        effective_area = (max_x - min_x) * (max_y - min_y)
    else:
        effective_area = sheet_size[0] * max_y if max_y > 0 else sheet_size[0] * sheet_size[1]
    
    if effective_area > 0:
        utilization = (total_area / effective_area) * 100
        return utilization
    return 0.0

def parallel_placement_with_restart(
    agents: List[IAGIAgent],
    sheet_size: Tuple[float, float],
    collision_detector: CollisionDetector,
    dynamics: GravitationalDynamics,
    min_gap: float,
    defect_zones: Optional[List[PolygonShape]] = None,
    time_limit: float = 300.0,
    restarts: int = 3
) -> List[IAGIAgent]:
    """
    Параллельный алгоритм размещения с многозапусковой стратегией
    
    :param restarts: Количество повторных запусков для поиска глобального оптимума
    :return: Лучшее найденное размещение
    """
    best_agents = None
    best_utilization = 0.0
    total_time_used = 0.0
    
    print(f"Запуск многозапусковой стратегии ({restarts} запусков)...")
    
    for restart in range(restarts):
        start_time = time.time()
        remaining_time = time_limit - total_time_used
        
        if remaining_time <= 0:
            print("Превышено общее время. Прекращение запусков.")
            break
        
        print(f"\n=== Запуск {restart + 1}/{restarts} ===")
        
        # Создание копий агентов для независимого запуска
        agents_copy = [agent.clone() for agent in agents]
        
        # Запуск параллельного размещения
        result_agents = parallel_placement(
            agents_copy,
            sheet_size,
            collision_detector,
            dynamics,
            min_gap,
            defect_zones,
            time_limit=remaining_time / (restarts - restart)
        )
        
        # Расчет утилизации для текущего запуска
        utilization = calculate_material_utilization(result_agents, sheet_size)
        elapsed = time.time() - start_time
        total_time_used += elapsed
        
        print(f"Запуск {restart + 1}: утилизация={utilization:.2f}%, время={elapsed:.2f}с")
        
        # Обновление лучшего решения
        if utilization > best_utilization:
            best_utilization = utilization
            best_agents = result_agents
        
        # Если осталось мало времени, прекращаем запуски
        if total_time_used > time_limit * 0.9:
            print("Достигнут лимит времени для многозапусковой стратегии.")
            break
    
    print(f"\nЛучший результат из {restarts} запусков: утилизация={best_utilization:.2f}%")
    return best_agents