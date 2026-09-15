import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional, Any
from core.agent import IAGIAgent
from core.geometry import PolygonShape
from core.collision import CollisionDetector
from core.dynamics import GravitationalDynamics
from core.optimizer import PackingOptimizer
from algorithms.sequential import sequential_placement
from algorithms.parallel import parallel_placement

class HybridGeneticIAGI:
    """
    Гибридный генетический алгоритм с ИАГИ-декодером (ГА-ИАГИ)
    
    Объединяет глобальный поиск с помощью генетического алгоритма
    и локальное уточнение с помощью физически корректного размещения ИАГИ.
    """
    
    def __init__(self, 
                 shapes: List[PolygonShape],
                 sheet_size: Tuple[float, float],
                 min_gap: float = 1.0,
                 population_size: int = 30,
                 generations: int = 50,
                 crossover_rate: float = 0.85,
                 mutation_rate: float = 0.15,
                 elite_size: int = 3,
                 time_limit: float = 300.0):
        """
        Инициализация гибридного ГА-ИАГИ
        
        :param shapes: Список фигур для размещения
        :param sheet_size: Размеры листа (ширина, высота)
        :param min_gap: Минимальный технологический зазор
        :param population_size: Размер популяции
        :param generations: Максимальное число поколений
        :param crossover_rate: Вероятность кроссовера
        :param mutation_rate: Вероятность мутации
        :param elite_size: Количество элитных особей для сохранения
        :param time_limit: Максимальное время работы в секундах
        """
        self.shapes = shapes
        self.n_shapes = len(shapes)
        self.sheet_size = sheet_size
        self.min_gap = min_gap
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.time_limit = time_limit
        
        # Настройки ИАГИ-декодера
        self.iagi_config = {
            'sheet_size': sheet_size,
            'min_gap': min_gap,
            'time_limit': 10.0,  # Максимальное время на один декодинг
            'use_sequential': True,
            'stabilization_enabled': True
        }
        
        # Инициализация компонентов для ИАГИ
        self.collision_detector = CollisionDetector(resolution=1.0)
        self.dynamics = GravitationalDynamics(
            gravity_strength=9.8,
            damping_linear=0.3,
            damping_angular=0.2
        )
        
        # Для хранения истории
        self.best_fitness_history = []
        self.average_fitness_history = []
        
    def create_initial_population(self) -> List[Dict[str, Any]]:
        """
        Создание начальной популяции хромосом
        
        Каждая хромосома содержит:
        - permutation: перестановка индексов фигур
        - angles: вектор углов поворота для каждой фигуры
        """
        population = []
        
        for _ in range(self.population_size):
            # Случайная перестановка индексов фигур
            permutation = list(range(self.n_shapes))
            random.shuffle(permutation)
            
            # Случайные углы (с учетом возможных ограничений)
            angles = []
            for shape in self.shapes:
                # Если есть ограничения на ориентацию
                if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                    angle = random.choice(shape.allowed_angles)
                else:
                    angle = random.uniform(0, 360)
                angles.append(angle)
            
            chromosome = {
                'permutation': permutation,
                'angles': angles,
                'fitness': 0.0,
                'placement': None  # Будет заполнено после декодирования
            }
            population.append(chromosome)
        
        return population
    
    def decode_chromosome(self, chromosome: Dict[str, Any]) -> Tuple[float, List[IAGIAgent]]:
        """
        Декодирование хромосомы с помощью ИАГИ
        
        :param chromosome: Хромосома с перестановкой и углами
        :return: (fitness, placement) - значение фитнеса и размещение
        """
        # Создание агентов в порядке, заданном перестановкой
        agents = []
        for idx in chromosome['permutation']:
            shape = self.shapes[idx]
            
            # Определение начальной позиции (сверху листа)
            init_position = np.array([
                self.sheet_size[0] / 2.0,
                self.sheet_size[1] + shape.get_bounding_box()[3] - shape.get_bounding_box()[1]
            ])
            
            # Создание агента
            agent = IAGIAgent(
                shape=shape,
                position=init_position,
                angle=chromosome['angles'][idx],
                priority=1  # Все фигуры в одном приоритете для ГА
            )
            agent.min_gap = self.min_gap
            
            # Установка ограничений на ориентацию, если они есть
            if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                agent.orientation_constraints = shape.allowed_angles
            
            agents.append(agent)
        
        # Запуск последовательного ИАГИ
        result_agents = sequential_placement(
            agents,
            self.sheet_size,
            self.collision_detector,
            self.dynamics,
            self.min_gap,
            time_limit=self.iagi_config['time_limit']
        )
        
        # Расчет фитнеса (коэффициент использования материала)
        fitness = self.calculate_fitness(result_agents)
        
        return fitness, result_agents
    
    def calculate_fitness(self, agents: List[IAGIAgent]) -> float:
        """
        Расчет фитнеса для размещения
        
        :param agents: Список агентов с финальным размещением
        :return: Значение фитнеса (коэффициент использования материала в %)
        """
        # Расчет общей площади фигур
        total_area = sum(agent.shape.area for agent in agents)
        
        # Расчет использованной высоты листа
        max_y = 0.0
        for agent in agents:
            transformed_shape = agent.get_transformed_shape()
            _, _, _, y_max = transformed_shape.get_bounding_box()
            max_y = max(max_y, y_max)
        
        # Эффективная площадь использования
        effective_area = self.sheet_size[0] * max_y
        sheet_area = self.sheet_size[0] * self.sheet_size[1]
        
        if effective_area > 0:
            utilization = (total_area / effective_area) * 100
        else:
            utilization = 0.0
        
        # Проверка коллизий (штраф за недопустимые решения)
        collision_penalty = 0.0
        for i, agent1 in enumerate(agents):
            shape1 = agent1.get_transformed_shape()
            for j in range(i + 1, len(agents)):
                agent2 = agents[j]
                shape2 = agent2.get_transformed_shape()
                
                if self.collision_detector.check_collision(shape1, shape2, self.min_gap):
                    # Штраф за каждую коллизию
                    collision_penalty += 5.0
        
        # Итоговый фитнес с учетом штрафов
        fitness = utilization - collision_penalty
        
        return fitness
    
    def selection(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Селекция особей для следующего поколения
        
        :param population: Текущая популяция
        :return: Отобранные особи
        """
        # Сортировка по фитнесу
        sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)
        
        # Элитизм - сохранение лучших особей
        new_population = sorted_population[:self.elite_size]
        
        # Турнирный отбор для остальных мест
        while len(new_population) < self.population_size:
            # Выбираем двух случайных кандидатов
            candidates = random.sample(sorted_population, 2)
            # Выбираем лучшего из двух
            winner = max(candidates, key=lambda x: x['fitness'])
            new_population.append(winner)
        
        return new_population
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Кроссовер (PMX - Partially Mapped Crossover) для перестановок
        
        :param parent1, parent2: Родительские хромосомы
        :return: Потомок после кроссовера
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # PMX для перестановок
        size = self.n_shapes
        child_permutation = [-1] * size
        
        # Выбираем сегмент для копирования из первого родителя
        start, end = sorted(random.sample(range(size), 2))
        
        # Копируем сегмент из первого родителя
        child_permutation[start:end+1] = parent1['permutation'][start:end+1]
        
        # Сопоставление для оставшихся позиций
        mapping = {}
        for i in range(start, end+1):
            mapping[parent2['permutation'][i]] = parent1['permutation'][i]
        
        # Заполнение оставшихся позиций
        for i in range(size):
            if i < start or i > end:
                value = parent2['permutation'][i]
                # Разрешение конфликтов с помощью mapping
                while value in child_permutation:
                    value = mapping.get(value, value)
                child_permutation[i] = value
        
        # Кроссовер для углов (арифметический)
        child_angles = []
        for i in range(size):
            if random.random() < 0.5:
                child_angles.append(parent1['angles'][i])
            else:
                child_angles.append(parent2['angles'][i])
        
        # Создание потомка
        child = {
            'permutation': child_permutation,
            'angles': child_angles,
            'fitness': 0.0,
            'placement': None
        }
        
        return child
    
    def mutate(self, chromosome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Мутация хромосомы
        
        :param chromosome: Исходная хромосома
        :return: Мутировавшая хромосома
        """
        mutated = chromosome.copy()
        
        # Мутация перестановки (обмен двух случайных позиций)
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n_shapes), 2)
            mutated['permutation'][i], mutated['permutation'][j] = \
                mutated['permutation'][j], mutated['permutation'][i]
        
        # Мутация углов
        for i in range(self.n_shapes):
            if random.random() < self.mutation_rate:
                shape = self.shapes[mutated['permutation'][i]]
                
                # Если есть ограничения на ориентацию
                if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                    mutated['angles'][i] = random.choice(shape.allowed_angles)
                else:
                    # Небольшое случайное изменение угла
                    delta = random.uniform(-15.0, 15.0)
                    mutated['angles'][i] = (mutated['angles'][i] + delta) % 360
        
        return mutated
    
    def optimize(self) -> Dict[str, Any]:
        """
        Основной метод оптимизации с помощью ГА-ИАГИ
        
        :return: Лучшее найденное решение
        """
        start_time = time.time()
        best_solution = None
        best_fitness = -float('inf')
        
        # Создание начальной популяции
        population = self.create_initial_population()
        
        # Декодирование и оценка начальной популяции
        for i, chromosome in enumerate(population):
            fitness, placement = self.decode_chromosome(chromosome)
            chromosome['fitness'] = fitness
            chromosome['placement'] = placement
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = chromosome.copy()
            
            # Проверка лимита времени
            if time.time() - start_time > self.time_limit:
                print(f"Достигнут лимит времени на инициализацию ({self.time_limit}с)")
                return best_solution
        
        # Эволюционный цикл
        for generation in range(self.generations):
            generation_start = time.time()
            
            # Селекция
            selected = self.selection(population)
            
            # Создание новой популяции через кроссовер и мутацию
            new_population = selected[:self.elite_size]  # Сохраняем элиту
            
            while len(new_population) < self.population_size:
                # Выбираем двух родителей
                parent1, parent2 = random.sample(selected, 2)
                
                # Кроссовер
                child = self.crossover(parent1, parent2)
                
                # Мутация
                child = self.mutate(child)
                
                # Декодирование и оценка
                fitness, placement = self.decode_chromosome(child)
                child['fitness'] = fitness
                child['placement'] = placement
                
                new_population.append(child)
                
                # Сохранение лучшего решения
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = child.copy()
            
            # Обновление популяции
            population = new_population
            
            # Сбор статистики
            avg_fitness = sum(ch['fitness'] for ch in population) / self.population_size
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(avg_fitness)
            
            # Логирование прогресса
            elapsed = time.time() - start_time
            gen_time = time.time() - generation_start
            print(f"Поколение {generation+1}/{self.generations}: "
                  f"лучший фитнес={best_fitness:.2f}%, "
                  f"средний фитнес={avg_fitness:.2f}%, "
                  f"время поколения={gen_time:.2f}с")
            
            # Проверка лимита времени
            if time.time() - start_time > self.time_limit:
                print(f"Достигнут лимит времени ({self.time_limit}с). Остановка на поколении {generation+1}.")
                break
        
        total_time = time.time() - start_time
        print(f"ГА-ИАГИ завершил работу за {total_time:.2f} секунд")
        print(f"Лучший результат: {best_fitness:.2f}% использования материала")
        
        return best_solution

class HybridPSOIAGI:
    """
    Гибридный алгоритм роя частиц с ИАГИ-декодером (PSO-ИАГИ)
    
    Объединяет глобальный поиск с помощью алгоритма роя частиц
    и локальное уточнение с помощью физически корректного размещения ИАГИ.
    """
    
    def __init__(self, 
                 shapes: List[PolygonShape],
                 sheet_size: Tuple[float, float],
                 min_gap: float = 1.0,
                 swarm_size: int = 30,
                 max_iterations: int = 100,
                 inertia_weight: float = 0.7,
                 cognitive_coeff: float = 1.5,
                 social_coeff: float = 1.5,
                 time_limit: float = 300.0):
        """
        Инициализация гибридного PSO-ИАГИ
        
        :param shapes: Список фигур для размещения
        :param sheet_size: Размеры листа (ширина, высота)
        :param min_gap: Минимальный технологический зазор
        :param swarm_size: Размер роя частиц
        :param max_iterations: Максимальное число итераций
        :param inertia_weight: Коэффициент инерции
        :param cognitive_coeff: Когнитивный коэффициент
        :param social_coeff: Социальный коэффициент
        :param time_limit: Максимальное время работы в секундах
        """
        self.shapes = shapes
        self.n_shapes = len(shapes)
        self.sheet_size = sheet_size
        self.min_gap = min_gap
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.time_limit = time_limit
        
        # Настройки ИАГИ-декодера
        self.iagi_config = {
            'sheet_size': sheet_size,
            'min_gap': min_gap,
            'time_limit': 8.0,  # Максимальное время на один декодинг
            'use_sequential': False,  # Используем параллельный режим для PSO
            'stabilization_enabled': True
        }
        
        # Инициализация компонентов для ИАГИ
        self.collision_detector = CollisionDetector(resolution=1.0)
        self.dynamics = GravitationalDynamics(
            gravity_strength=9.8,
            damping_linear=0.3,
            damping_angular=0.2
        )
        
        # Для хранения истории
        self.best_fitness_history = []
        self.average_fitness_history = []
        
    def create_initial_swarm(self) -> List[Dict[str, Any]]:
        """
        Создание начального роя частиц
        
        Каждая частица представляет собой:
        - position: вектор позиций и углов для всех фигур
        - velocity: вектор скоростей
        - best_position: лучшая позиция частицы
        - best_fitness: лучший фитнес частицы
        """
        swarm = []
        
        for _ in range(self.swarm_size):
            # Позиция: [x1, y1, angle1, x2, y2, angle2, ...]
            position = []
            velocity = []
            
            for shape in self.shapes:
                # Случайные начальные позиции в пределах листа
                x = random.uniform(0, self.sheet_size[0])
                y = random.uniform(0, self.sheet_size[1])
                
                # Случайные углы (с учетом ограничений)
                if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                    angle = random.choice(shape.allowed_angles)
                else:
                    angle = random.uniform(0, 360)
                
                position.extend([x, y, angle])
                
                # Случайные начальные скорости
                vx = random.uniform(-10, 10)
                vy = random.uniform(-10, 10)
                v_angle = random.uniform(-10, 10)
                velocity.extend([vx, vy, v_angle])
            
            particle = {
                'position': np.array(position),
                'velocity': np.array(velocity),
                'best_position': np.array(position),
                'best_fitness': -float('inf'),
                'placement': None
            }
            swarm.append(particle)
        
        return swarm
    
    def decode_particle(self, position: np.ndarray) -> Tuple[float, List[IAGIAgent]]:
        """
        Декодирование позиции частицы с помощью ИАГИ
        
        :param position: Вектор позиций и углов
        :return: (fitness, placement) - значение фитнеса и размещение
        """
        # Создание агентов
        agents = []
        for i, shape in enumerate(self.shapes):
            idx = i * 3
            x, y, angle = position[idx], position[idx+1], position[idx+2]
            
            # Создание агента
            agent = IAGIAgent(
                shape=shape,
                position=np.array([x, y]),
                angle=angle,
                priority=1  # Все фигуры в одном приоритете для PSO
            )
            agent.min_gap = self.min_gap
            
            # Установка ограничений на ориентацию
            if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                agent.orientation_constraints = shape.allowed_angles
            
            agents.append(agent)
        
        # Запуск параллельного ИАГИ
        result_agents = parallel_placement(
            agents,
            self.sheet_size,
            self.collision_detector,
            self.dynamics,
            self.min_gap,
            time_limit=self.iagi_config['time_limit']
        )
        
        # Расчет фитнеса
        fitness = self.calculate_fitness(result_agents)
        
        return fitness, result_agents
    
    def calculate_fitness(self, agents: List[IAGIAgent]) -> float:
        """
        Расчет фитнеса для размещения (аналогичен ГА-ИАГИ)
        """
        # Расчет общей площади фигур
        total_area = sum(agent.shape.area for agent in agents)
        
        # Расчет использованной высоты листа
        max_y = 0.0
        for agent in agents:
            transformed_shape = agent.get_transformed_shape()
            _, _, _, y_max = transformed_shape.get_bounding_box()
            max_y = max(max_y, y_max)
        
        # Эффективная площадь использования
        effective_area = self.sheet_size[0] * max_y
        sheet_area = self.sheet_size[0] * self.sheet_size[1]
        
        if effective_area > 0:
            utilization = (total_area / effective_area) * 100
        else:
            utilization = 0.0
        
        # Проверка коллизий (штраф за недопустимые решения)
        collision_penalty = 0.0
        for i, agent1 in enumerate(agents):
            shape1 = agent1.get_transformed_shape()
            for j in range(i + 1, len(agents)):
                agent2 = agents[j]
                shape2 = agent2.get_transformed_shape()
                
                if self.collision_detector.check_collision(shape1, shape2, self.min_gap):
                    # Штраф за каждую коллизию
                    collision_penalty += 5.0
        
        # Итоговый фитнес с учетом штрафов
        fitness = utilization - collision_penalty
        
        return fitness
    
    def update_velocity(self, particle: Dict[str, Any], global_best_position: np.ndarray) -> np.ndarray:
        """
        Обновление скорости частицы
        
        :param particle: Текущая частица
        :param global_best_position: Глобальная лучшая позиция
        :return: Новый вектор скорости
        """
        r1 = random.random()
        r2 = random.random()
        
        cognitive_component = self.cognitive_coeff * r1 * (particle['best_position'] - particle['position'])
        social_component = self.social_coeff * r2 * (global_best_position - particle['position'])
        
        new_velocity = (self.inertia_weight * particle['velocity'] + 
                        cognitive_component + 
                        social_component)
        
        # Ограничение скорости для предотвращения взрывного поведения
        max_velocity = 20.0
        velocity_norm = np.linalg.norm(new_velocity)
        if velocity_norm > max_velocity:
            new_velocity = new_velocity * (max_velocity / velocity_norm)
        
        return new_velocity
    
    def update_position(self, particle: Dict[str, Any]) -> np.ndarray:
        """
        Обновление позиции частицы с учетом границ листа
        
        :param particle: Текущая частица
        :return: Новая позиция
        """
        new_position = particle['position'] + particle['velocity']
        
        # Обработка границ для координат x и y
        for i in range(self.n_shapes):
            idx = i * 3
            # Ограничение x в пределах листа
            new_position[idx] = max(0, min(self.sheet_size[0], new_position[idx]))
            # Ограничение y в пределах листа
            new_position[idx+1] = max(0, min(self.sheet_size[1], new_position[idx+1]))
            # Нормализация угла
            new_position[idx+2] = new_position[idx+2] % 360
            
            # Учет ограничений на ориентацию, если они есть
            shape = self.shapes[i]
            if hasattr(shape, 'allowed_angles') and shape.allowed_angles:
                # Найти ближайший допустимый угол
                allowed_angles = shape.allowed_angles
                current_angle = new_position[idx+2]
                diffs = [(angle - current_angle) % 360 for angle in allowed_angles]
                min_diff_idx = np.argmin(np.abs(diffs))
                new_position[idx+2] = allowed_angles[min_diff_idx]
        
        return new_position
    
    def optimize(self) -> Dict[str, Any]:
        """
        Основной метод оптимизации с помощью PSO-ИАГИ
        
        :return: Лучшее найденное решение
        """
        start_time = time.time()
        best_global_fitness = -float('inf')
        best_global_position = None
        best_global_placement = None
        
        # Создание начального роя
        swarm = self.create_initial_swarm()
        
        # Инициализация частиц
        for particle in swarm:
            fitness, placement = self.decode_particle(particle['position'])
            particle['best_fitness'] = fitness
            particle['best_position'] = np.copy(particle['position'])
            particle['placement'] = placement
            
            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_position = np.copy(particle['position'])
                best_global_placement = placement
        
        # Основной цикл PSO
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            for particle in swarm:
                # Обновление скорости и позиции
                particle['velocity'] = self.update_velocity(particle, best_global_position)
                particle['position'] = self.update_position(particle)
                
                # Декодирование новой позиции
                fitness, placement = self.decode_particle(particle['position'])
                
                # Обновление личного лучшего
                if fitness > particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = np.copy(particle['position'])
                    particle['placement'] = placement
                
                # Обновление глобального лучшего
                if fitness > best_global_fitness:
                    best_global_fitness = fitness
                    best_global_position = np.copy(particle['position'])
                    best_global_placement = placement
            
            # Сбор статистики
            avg_fitness = sum(p['best_fitness'] for p in swarm) / self.swarm_size
            self.best_fitness_history.append(best_global_fitness)
            self.average_fitness_history.append(avg_fitness)
            
            # Логирование прогресса
            elapsed = time.time() - start_time
            iter_time = time.time() - iteration_start
            print(f"Итерация {iteration+1}/{self.max_iterations}: "
                  f"лучший фитнес={best_global_fitness:.2f}%, "
                  f"средний фитнес={avg_fitness:.2f}%, "
                  f"время итерации={iter_time:.2f}с")
            
            # Проверка лимита времени
            if time.time() - start_time > self.time_limit:
                print(f"Достигнут лимит времени ({self.time_limit}с). Остановка на итерации {iteration+1}.")
                break
        
        # Формирование лучшего решения
        best_solution = {
            'position': best_global_position,
            'fitness': best_global_fitness,
            'placement': best_global_placement,
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history
        }
        
        total_time = time.time() - start_time
        print(f"PSO-ИАГИ завершил работу за {total_time:.2f} секунд")
        print(f"Лучший результат: {best_global_fitness:.2f}% использования материала")
        
        return best_solution

def hybrid_optimization_factory(
    algorithm_type: str,
    shapes: List[PolygonShape],
    sheet_size: Tuple[float, float],
    min_gap: float = 1.0,
    time_limit: float = 300.0,
    **kwargs
) -> Any:
    """
    Фабрика для создания гибридных оптимизаторов
    
    :param algorithm_type: Тип алгоритма ('GA-IAGI' или 'PSO-IAGI')
    :param shapes: Список фигур для размещения
    :param sheet_size: Размеры листа (ширина, высота)
    :param min_gap: Минимальный технологический зазор
    :param time_limit: Максимальное время работы в секундах
    :param kwargs: Дополнительные параметры для конкретного алгоритма
    :return: Экземпляр оптимизатора
    """
    if algorithm_type.upper() == 'GA-IAGI':
        return HybridGeneticIAGI(
            shapes=shapes,
            sheet_size=sheet_size,
            min_gap=min_gap,
            time_limit=time_limit,
            **kwargs
        )
    elif algorithm_type.upper() == 'PSO-IAGI':
        return HybridPSOIAGI(
            shapes=shapes,
            sheet_size=sheet_size,
            min_gap=min_gap,
            time_limit=time_limit,
            **kwargs
        )
    else:
        raise ValueError(f"Неизвестный тип гибридного алгоритма: {algorithm_type}. "
                         f"Поддерживаются только 'GA-IAGI' и 'PSO-IAGI'")