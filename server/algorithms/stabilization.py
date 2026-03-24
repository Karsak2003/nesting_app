import numpy as np
from typing import List

def check_energy_stagnation(energy_history: List[float], stagnation_steps: int, energy_threshold: float) -> bool:
    if len(energy_history) < stagnation_steps:
        return False

    recent_energy = energy_history[-stagnation_steps:]

    min_energy = min(recent_energy)
    max_energy = max(recent_energy)
    energy_range = max_energy - min_energy

    return energy_range < energy_threshold

def apply_stabilization(agents, collision_detector, dynamics, min_gap):
    """
    Применение комплексных механизмов стабилизации
    """
    print("Применение механизмов стабилизации...")
    
    # 1. Адаптивное демпфирование
    for agent in agents:
        if not agent.is_frozen:
            velocity_norm = np.linalg.norm(agent.velocity)
            if velocity_norm < 1.0:  # Низкая скорость
                dynamics.damping_linear = min(0.95, dynamics.damping_linear * 1.2)
                dynamics.damping_angular = min(0.95, dynamics.damping_angular * 1.2)
    
    max_stabilization_iterations = 10
    for stabilization_iter in range(max_stabilization_iterations):
        collision_map = {}
        has_collisions = False
        
        for i, agent1 in enumerate(agents):
            if agent1.is_frozen:
                continue
                
            shape1 = agent1.get_transformed_shape()
            for j, agent2 in enumerate(agents):
                if i >= j or agent2.is_frozen:
                    continue
                    
                shape2 = agent2.get_transformed_shape()
                if collision_detector.check_collision(shape1, shape2, min_gap):
                    has_collisions = True
                    distance, cp1, cp2, normal = collision_detector.calculate_min_distance(
                        shape1, shape2, min_gap
                    )
                    
                    if distance < 0:  # Есть столкновение
                        penetration_depth = -distance
                        collision_map.setdefault(i, []).append((j, penetration_depth, normal))
        
        if not has_collisions:
            break
        
        # Разрешение коллизий
        processed_pairs = set()
        
        for i, collisions in collision_map.items():
            agent = agents[i]
            
            for j, penetration_depth, normal in collisions:
                pair_key = (min(i, j), max(i, j))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                agent2 = agents[j]
                
                # Смещение пропорционально массам агентов
                total_mass = agent.mass + agent2.mass
                if total_mass > 0:
                    ratio1 = agent.mass / total_mass
                    ratio2 = agent2.mass / total_mass
                else:
                    ratio1 = ratio2 = 0.5
                
                displacement_factor = 0.9 + 0.1 * (stabilization_iter + 1) / max_stabilization_iterations
                
                displacement = normal * (penetration_depth + min_gap) * displacement_factor
                
                agent.position += displacement * ratio2
                agent2.position -= displacement * ratio1
    
    # 3. Перезапуск при стагнации (многозапусковой подход)
    total_energy = sum(np.linalg.norm(agent.velocity)**2 + agent.position[1] for agent in agents 
                      if not agent.is_frozen)
    
    if hasattr(apply_stabilization, 'prev_energy') and hasattr(apply_stabilization, 'stagnation_count'):
        energy_diff = abs(total_energy - apply_stabilization.prev_energy)
        if energy_diff < 0.1:  # Стагнация
            apply_stabilization.stagnation_count += 1
            if apply_stabilization.stagnation_count > 10:
                print("Обнаружена стагнация. Применение случайного смещения...")
                for agent in agents:
                    if not agent.is_frozen and np.random.random() < 0.3:
                        angle = np.random.uniform(0, 2*np.pi)
                        displacement = np.random.uniform(0, 5.0)  # мм
                        agent.position += np.array([
                            displacement * np.cos(angle),
                            displacement * np.sin(angle)
                        ])
                apply_stabilization.stagnation_count = 0
        else:
            apply_stabilization.stagnation_count = 0
    else:
        apply_stabilization.stagnation_count = 0
    
    apply_stabilization.prev_energy = total_energy
    
    return agents