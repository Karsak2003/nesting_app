import time
import numpy as np
import logging
from typing import Tuple, Any, Optional, Callable

logger = logging.getLogger(__name__)

def sequential_placement(
    agents: list, 
    sheet_size: Tuple[float, float], 
    collision_detector: Any, 
    dynamics: Any, 
    min_gap: float, 
    defect_zones: Optional[list] = None, 
    time_limit: float = 600, 
    progress_callback: Optional[Callable[[int, list, float, float], None]] = None
) -> list:
    """
    Последовательный алгоритм размещения с учетом приоритетов
    
    Args:
        agents: список агентов для размещения
        sheet_size: размеры листа (width, height)
        collision_detector: детектор столкновений
        dynamics: физическая модель динамики
        min_gap: минимальный зазор между фигурами
        defect_zones: список дефектных зон
        time_limit: лимит времени выполнения в секундах
        progress_callback: функция обратного вызова (progress, agents, utilization)
    """
    if defect_zones is None:
        defect_zones = []
        
    start_time = time.time()
    sorted_agents = sorted(agents, key=lambda x: x.priority, reverse=True)
    
    # Группируем агентов по приоритетам
    priority_groups = {}
    for agent in sorted_agents:
        priority_groups.setdefault(agent.priority, []).append(agent)
    
    total_groups = len(priority_groups)
    completed_groups = 0
    
    # Обработка каждой группы приоритетов
    for priority in sorted(priority_groups.keys(), reverse=True):
        group = priority_groups[priority]
        logger.info(f"=== Размещение группы приоритета {priority} (фигур: {len(group)}) ===")
        print(f"Размещение группы приоритета {priority} (фигур: {len(group)})")
        
        # Логирование начальных позиций фигур
        for agent in group:
            transformed = agent.get_transformed_shape()
            bbox = transformed.get_bounding_box()
            logger.info(f"  Начальная позиция {agent.shape.name}: pos=({agent.position[0]:.2f}, {agent.position[1]:.2f}), "
                       f"bbox_y={bbox[1]:.2f}..{bbox[3]:.2f}, высота={bbox[3]-bbox[1]:.2f} мм")
        
        # Активируем агентов этой группы
        for agent in group:
            agent.is_frozen = False
        
        # Флаг для отслеживания успешного завершения группы
        group_completed = False
        
        # Основной цикл симуляции для текущей группы
        stable = False
        iteration = 0
        max_iterations = 50000
        
        logger.debug(f"Начало цикла размещения для группы приоритета {priority}, stable={stable}, iteration={iteration}")
        _exit_reason = None
        while not stable and (time.time() - start_time) < time_limit and iteration < max_iterations:
            logger.debug(f"Вход в цикл: iteration={iteration}, stable={stable}")
            stable = True
            max_velocity = 0.0
            total_force_magnitude = 0.0
            agent_details = []
            
            if iteration == 0:
                logger.info(f"Начало итераций для группы приоритета {priority}, time_limit={time_limit:.1f}с")
                print(f"  Начало итераций для группы приоритета {priority}")
            
            for agent in group:
                if agent.is_frozen:
                    continue
                    
                agent_transformed = agent.get_transformed_shape()
                transformed_shapes = [a.get_transformed_shape() for a in agents]
                
                nearby_indices = collision_detector.find_nearby_shapes(
                    agent_transformed, 
                    transformed_shapes, 
                    agent.perception_radius
                )
                neighbors = [agents[i] for i in nearby_indices if i < len(agents) and agents[i] is not agent]
                
                # Обновление знаний агента
                agent.update_beliefs(neighbors, sheet_size, defect_zones)
                
                # Расчет сил и моментов
                force, torque = agent.compute_intention(collision_detector, dynamics)
                force_magnitude = np.linalg.norm(force)
                total_force_magnitude += force_magnitude
                
                # Интегрирование уравнений движения
                prev_position = agent.position.copy()
                dynamics.verlet_integration(agent, force, torque, dt=0.08)  # Увеличен шаг для ускорения (было 0.05)
                
                # Проверка стабильности
                movement = np.linalg.norm(agent.position - prev_position)
                if movement > 0.1:  # мм за шаг
                    stable = False
                
                transformed_shape = agent.get_transformed_shape()
                bbox = transformed_shape.get_bounding_box()
                    
                velocity_norm = np.linalg.norm(agent.velocity)
                max_velocity = max(max_velocity, velocity_norm)

                if iteration % 20 == 0:
                    agent_details.append({
                        'name': agent.shape.name,
                        'position': (agent.position[0], agent.position[1]),
                        'movement': movement,
                        'velocity': velocity_norm,
                        'force': force_magnitude,
                        'bbox_y': (bbox[1], bbox[3]),
                        'height': bbox[3] - bbox[1]
                    })
                
                # Проверка столкновений
                transformed_shape = agent.get_transformed_shape()
                for other_agent in agents:
                    if other_agent is agent or other_agent.is_frozen:
                        continue
                        
                    other_shape = other_agent.get_transformed_shape()
                    if collision_detector.check_collision(transformed_shape, other_shape, min_gap):
                        stable = False
            
            iteration += 1
            if iteration % 20 == 0:        
                print(f"  Итерация {iteration}, max_velocity={max_velocity:.2f} мм/с")
                
                # Вызов callback для обновления прогресса
                if progress_callback:
                    # Расчет текущего прогресса и утилизации
                    progress = int((completed_groups + (iteration / max_iterations)) / total_groups * 100)
                    total_area = sum(a.shape.area for a in agents)
                    max_y = max(a.get_transformed_shape().get_bounding_box()[3] for a in agents)
                    effective_area = sheet_size[0] * max_y if max_y > 0 else total_area
                    utilization = (total_area / effective_area) * 100 if effective_area > 0 else 0.0
                    
                    # <-- ДОБАВЛЕНО: Расчет текущей энергии системы для активных агентов группы
                    current_energy = sum(
                        0.5 * a.mass * np.linalg.norm(a.velocity)**2 + 
                        0.5 * a.moment_of_inertia * (a.angular_velocity**2) + 
                        a.mass * 9.8 * a.position[1]
                        for a in group if not a.is_frozen
                    )
                    
                    # Передаем 4 аргумента: progress, agents, utilization, energy
                    progress_callback(progress, agents, utilization, current_energy)
                
            # Проверка стабилизации группы
            if stable or max_velocity < 0.5:
                print(f"  Группа приоритета {priority} стабилизирована за {iteration} итераций")
                # Замораживаем агентов этой группы
                for agent in group:
                    agent.is_frozen = True
                completed_groups += 1
                break
    
    elapsed_time = time.time() - start_time
    print(f"Последовательное размещение завершено за {elapsed_time:.2f} секунд")
    
    # Финальный вызов callback
    if progress_callback:
        total_area = sum(a.shape.area for a in agents)
        max_y = max(a.get_transformed_shape().get_bounding_box()[3] for a in agents)
        effective_area = sheet_size[0] * max_y if max_y > 0 else total_area
        utilization = (total_area / effective_area) * 100 if effective_area > 0 else 0.0
        
        final_energy = sum(
            0.5 * a.mass * np.linalg.norm(a.velocity)**2 + 
            0.5 * a.moment_of_inertia * (a.angular_velocity**2) + 
            a.mass * 9.8 * a.position[1]
            for a in agents if not a.is_frozen
        )
        
        progress_callback(100, agents, utilization, final_energy)

    return agents