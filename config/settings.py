import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np

class SystemConfig:
    """
    Класс для управления конфигурацией системы раскроя-упаковки на основе ИАГИ
    
    Реализует требования глав 2-3 диссертации:
    - Интеграция технологических ограничений (раздел 2.5)
    - Параметры динамической системы (разделы 2.3-2.4)
    - Настройки алгоритмов размещения (раздел 3.3)
    - Параметры стабилизации и предотвращения залипания (раздел 3.5)
    - Профили конфигурации для разных отраслей (раздел 3.6.4)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Инициализация конфигурации системы
        
        :param config_file: Путь к файлу конфигурации (YAML/JSON)
        """
        # Базовые параметры системы
        self.system = {
            'name': 'IAGI-Nesting-System',
            'version': '1.0.0',
            'debug_mode': False,
            'log_level': 'INFO',
            'log_file': 'logs/nesting_system.log',
            'max_execution_time': 600,  # секунд (10 минут)
            'num_threads': max(1, os.cpu_count() - 1),  # Оставить 1 ядро для системы
            'random_seed': 42,
            'precision': np.float32
        }
        
        # Параметры геометрического представления (раздел 2.1)
        self.geometry = {
            'polygonization': {
                'tolerance': 0.1,        # мм (точность полигонализации NURBS)
                'max_vertices': 200,    # Максимальное число вершин на фигуру
                'adaptive': True,       # Использовать адаптивную полигонализацию
                'preserve_topology': True  # Сохранять топологию (отверстия)
            },
            'distance_field': {
                'resolution': 0.5,      # мм (разрешение SDF)
                'padding': 5.0,         # мм (отступ от границ фигуры)
                'use_adaptive': True    # Адаптивное разрешение вблизи границ
            },
            'collision_detection': {
                'min_gap': 1.0,         # мм (минимальный технологический зазор)
                'bvh_max_objects_per_leaf': 4,
                'use_bvh': True,
                'sdf_enabled': True,
                'fallback_to_exact': True
            }
        }
        
        # Параметры динамической системы (разделы 2.3-2.4)
        self.dynamics = {
            'gravity': {
                'enabled': True,
                'strength': 9.8,        # м/с² (ускорение свободного падения)
                'direction': [0.0, -1.0]  # Направление гравитации (вниз)
            },
            'repulsion': {
                'enabled': True,
                'strength': 100.0,      # Базовая сила отталкивания
                'exponent': 2.5,        # Степень в потенциале отталкивания
                'range_factor': 2.0,    # Радиус действия (кратно минимальному зазору)
                'regularization_epsilon': 0.1  # мм (регуляризация для численной устойчивости)
            },
            'damping': {
                'enabled': True,
                'linear': 0.3,          # Коэффициент линейного демпфирования
                'angular': 0.2,         # Коэффициент углового демпфирования
                'adaptive': True,       # Адаптивное демпфирование (раздел 3.5.2)
                'adaptive_threshold': 1.0  # мм/с (порог для активации адаптивного демпфирования)
            },
            'integration': {
                'method': 'verlet',     # Метод интегрирования
                'time_step': 0.01,      # с (базовый шаг интегрирования)
                'adaptive_step': True,  # Адаптивный шаг интегрирования
                'max_step': 0.05,       # с (максимальный шаг)
                'min_step': 0.001       # с (минимальный шаг)
            }
        }
        
        # Параметры алгоритмов размещения (раздел 3.3)
        self.placement = {
            'mode': 'sequential',      # 'sequential', 'parallel', 'hybrid'
            'priority_strategy': 'area',  # 'area', 'complexity', 'manual'
            'stabilization_enabled': True,
            'stabilization': {
                'energy_threshold': 0.1,    # Порог изменения энергии для сходимости
                'max_stagnation_steps': 50,  # Максимальное число шагов при стагнации
                'relaxation_steps': 20,      # Число шагов для релаксации коллизий
                'restart_enabled': True,    # Перезапуск при стагнации
                'restart_count': 3,         # Число перезапусков
                'micro_displacement': 1.0   # мм (максимальное микросмещение)
            },
            'hybrid': {
                'ga_enabled': False,        # Использовать гибрид ГА-ИАГИ
                'pso_enabled': False,       # Использовать гибрид PSO-ИАГИ
                'population_size': 30,      # Размер популяции для ГА/PSO
                'generations': 50,          # Число поколений для ГА/PSO
                'crossover_rate': 0.85,     # Вероятность кроссовера для ГА
                'mutation_rate': 0.15,      # Вероятность мутации для ГА
                'inertia_weight': 0.7,      # Коэффициент инерции для PSO
                'cognitive_coeff': 1.5,     # Когнитивный коэффициент для PSO
                'social_coeff': 1.5         # Социальный коэффициент для PSO
            }
        }
        
        # Технологические ограничения (раздел 2.5)
        self.technological_constraints = {
            'cutting_technology': 'laser',  # 'laser', 'plasma', 'waterjet', 'default'
            'min_gaps': {
                'laser': 0.2,       # мм
                'plasma': 2.0,      # мм
                'waterjet': 1.0,    # мм
                'default': 1.0      # мм
            },
            'orientation_constraints': {
                'enabled': False,
                'default_angles': [0.0],  # По умолчанию только 0 градусов
                'anisotropic_materials': []  # Список ID анизотропных материалов
            },
            'defect_zones': {
                'enabled': False,
                'min_distance_factor': 1.5  # Коэффициент увеличения зазора до дефектов
            },
            'priority_rules': {
                'enabled': True,
                'default_priority': 1,
                'max_priority_level': 5
            },
            'standards': {
                'GOST_34029_2016': True,    # Требования к раскрою листовых материалов
                'GOST_R_56079_2014': True,  # Требования к дефектным зонам
                'ISO_10303_21': True,       # STEP формат
                'kerf_compensation': True   # Компенсация ширины реза
            }
        }
        
        # Профили конфигурации для разных отраслей (раздел 3.6.4)
        self.profiles = {
            'high_precision': {  # Авиация, космос
                'geometry': {
                    'polygonization': {'tolerance': 0.05},
                    'distance_field': {'resolution': 0.2}
                },
                'dynamics': {
                    'repulsion': {'strength': 150.0},
                    'damping': {'linear': 0.4, 'angular': 0.3}
                },
                'placement': {
                    'mode': 'sequential',
                    'stabilization_enabled': True
                },
                'technological_constraints': {
                    'cutting_technology': 'laser',
                    'min_gaps': {'default': 0.5}
                }
            },
            'medium_precision': {  # Машиностроение
                'geometry': {
                    'polygonization': {'tolerance': 0.1},
                    'distance_field': {'resolution': 0.5}
                },
                'dynamics': {
                    'repulsion': {'strength': 100.0},
                    'damping': {'linear': 0.3, 'angular': 0.2}
                },
                'placement': {
                    'mode': 'hybrid',
                    'stabilization_enabled': True
                },
                'technological_constraints': {
                    'cutting_technology': 'laser',
                    'min_gaps': {'default': 1.0}
                }
            },
            'low_precision': {  # Текстиль, картон
                'geometry': {
                    'polygonization': {'tolerance': 0.5},
                    'distance_field': {'resolution': 1.0}
                },
                'dynamics': {
                    'repulsion': {'strength': 50.0},
                    'damping': {'linear': 0.2, 'angular': 0.1}
                },
                'placement': {
                    'mode': 'parallel',
                    'stabilization_enabled': False
                },
                'technological_constraints': {
                    'cutting_technology': 'waterjet',
                    'min_gaps': {'default': 2.0}
                }
            }
        }
        
        # Пути к файлам и директориям
        self.paths = {
            'data_dir': 'data',
            'results_dir': 'results',
            'logs_dir': 'logs',
            'cache_dir': 'cache',
            'templates_dir': 'templates'
        }
        
        # Инициализация логирования
        self._setup_logging()
        
        # Загрузка конфигурации из файла, если указан
        if config_file:
            self.load_from_file(config_file)
        
        # Применение профиля по умолчанию
        self.apply_profile('medium_precision')
        
        self.logger.info("Конфигурация системы успешно инициализирована")
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        # Создание директорий для логов
        log_dir = os.path.dirname(self.system['log_file'])
        os.makedirs(log_dir, exist_ok=True)
        
        # Настройка форматтера
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Настройка обработчиков
        file_handler = logging.FileHandler(self.system['log_file'])
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.system['log_level']))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger('SystemConfig')
        self.logger.info("Логирование настроено")
    
    def apply_profile(self, profile_name: str):
        """
        Применение профиля конфигурации для конкретной отрасли
        
        :param profile_name: Имя профиля ('high_precision', 'medium_precision', 'low_precision')
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Неизвестный профиль конфигурации: {profile_name}. "
                           f"Доступные: {list(self.profiles.keys())}")
        
        profile = self.profiles[profile_name]
        self.logger.info(f"Применение профиля конфигурации: {profile_name}")
        
        # Рекурсивное обновление параметров из профиля
        self._update_config_from_profile(self, profile)
        
        # Установка точности полигонализации в зависимости от профиля
        if profile_name == 'high_precision':
            self.geometry['polygonization']['tolerance'] = 0.05
            self.geometry['distance_field']['resolution'] = 0.2
        elif profile_name == 'medium_precision':
            self.geometry['polygonization']['tolerance'] = 0.1
            self.geometry['distance_field']['resolution'] = 0.5
        elif profile_name == 'low_precision':
            self.geometry['polygonization']['tolerance'] = 0.5
            self.geometry['distance_field']['resolution'] = 1.0
        
        # Обновление минимального зазора в зависимости от технологии резки
        technology = self.technological_constraints['cutting_technology']
        if technology in self.technological_constraints['min_gaps']:
            self.geometry['collision_detection']['min_gap'] = \
                self.technological_constraints['min_gaps'][technology]
    
    def _update_config_from_profile(self, config_obj, profile_dict):
        """
        Рекурсивное обновление конфигурации из профиля
        
        :param config_obj: Объект конфигурации для обновления
        :param profile_dict: Словарь с параметрами профиля
        """
        for key, value in profile_dict.items():
            if hasattr(config_obj, key) and isinstance(getattr(config_obj, key), dict):
                self._update_config_from_profile(getattr(config_obj, key), value)
            elif isinstance(config_obj, dict) and key in config_obj:
                if isinstance(value, dict) and isinstance(config_obj[key], dict):
                    self._update_config_from_profile(config_obj[key], value)
                else:
                    config_obj[key] = value
    
    def set_technology(self, technology: str):
        """
        Установка технологии резки и соответствующих параметров
        
        :param technology: 'laser', 'plasma', 'waterjet' или 'default'
        """
        if technology not in self.technological_constraints['min_gaps']:
            raise ValueError(f"Неизвестная технология резки: {technology}. "
                           f"Доступные: {list(self.technological_constraints['min_gaps'].keys())}")
        
        self.technological_constraints['cutting_technology'] = technology
        self.geometry['collision_detection']['min_gap'] = \
            self.technological_constraints['min_gaps'][technology]
        
        self.logger.info(f"Установлена технология резки: {technology}, "
                        f"минимальный зазор: {self.geometry['collision_detection']['min_gap']} мм")
    
    def set_placement_mode(self, mode: str):
        """
        Установка режима размещения
        
        :param mode: 'sequential', 'parallel' или 'hybrid'
        """
        valid_modes = ['sequential', 'parallel', 'hybrid']
        if mode not in valid_modes:
            raise ValueError(f"Недопустимый режим размещения: {mode}. "
                           f"Допустимые значения: {valid_modes}")
        
        self.placement['mode'] = mode
        self.logger.info(f"Установлен режим размещения: {mode}")
    
    def get_current_profile(self) -> str:
        """
        Получение текущего профиля конфигурации
        
        :return: Имя текущего профиля
        """
        # Определение профиля по ключевым параметрам
        tolerance = self.geometry['polygonization']['tolerance']
        
        if tolerance <= 0.05:
            return 'high_precision'
        elif tolerance <= 0.1:
            return 'medium_precision'
        else:
            return 'low_precision'
    
    def load_from_file(self, config_file: str):
        """
        Загрузка конфигурации из файла
        
        :param config_file: Путь к файлу конфигурации (YAML/JSON)
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError("Поддерживаются только файлы форматов YAML и JSON")
            
            # Рекурсивное обновление параметров
            self._update_config_from_file(config_data)
            self.logger.info(f"Конфигурация успешно загружена из файла: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке конфигурации из файла {config_file}: {e}")
            raise
    
    def _update_config_from_file(self, config_data: Dict[str, Any]):
        """
        Рекурсивное обновление конфигурации из словаря
        
        :param config_data: Словарь с параметрами конфигурации
        """
        for section, params in config_data.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if isinstance(section_obj, dict):
                    self._update_dict_from_file(section_obj, params)
    
    def _update_dict_from_file(self, target_dict: Dict[str, Any], source_dict: Dict[str, Any]):
        """
        Рекурсивное обновление словаря конфигурации
        
        :param target_dict: Целевой словарь для обновления
        :param source_dict: Источник данных
        """
        for key, value in source_dict.items():
            if key in target_dict:
                if isinstance(value, dict) and isinstance(target_dict[key], dict):
                    self._update_dict_from_file(target_dict[key], value)
                else:
                    target_dict[key] = value
            else:
                self.logger.warning(f"Неизвестный параметр конфигурации: {key}")
    
    def save_to_file(self, config_file: str):
        """
        Сохранение конфигурации в файл
        
        :param config_file: Путь к файлу для сохранения (YAML/JSON)
        """
        try:
            # Сбор всех параметров конфигурации в словарь
            config_data = {
                'system': self.system,
                'geometry': self.geometry,
                'dynamics': self.dynamics,
                'placement': self.placement,
                'technological_constraints': self.technological_constraints,
                'profiles': self.profiles,
                'paths': self.paths
            }
            
            # Создание директории, если необходимо
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            # Сохранение в зависимости от формата файла
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            elif config_file.endswith('.json'):
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError("Поддерживаются только файлы форматов YAML и JSON")
            
            self.logger.info(f"Конфигурация успешно сохранена в файл: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении конфигурации в файл {config_file}: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Валидация конфигурации на корректность параметров
        
        :return: True если конфигурация валидна
        """
        valid = True
        
        # Проверка положительных параметров
        positive_params = [
            ('geometry.polygonization.tolerance', self.geometry['polygonization']['tolerance']),
            ('geometry.distance_field.resolution', self.geometry['distance_field']['resolution']),
            ('geometry.collision_detection.min_gap', self.geometry['collision_detection']['min_gap']),
            ('dynamics.gravity.strength', self.dynamics['gravity']['strength']),
            ('dynamics.repulsion.strength', self.dynamics['repulsion']['strength']),
            ('dynamics.repulsion.exponent', self.dynamics['repulsion']['exponent']),
            ('system.max_execution_time', self.system['max_execution_time']),
            ('system.num_threads', self.system['num_threads'])
        ]
        
        for param_name, value in positive_params:
            if value <= 0:
                self.logger.error(f"Некорректное значение параметра {param_name}: {value}. "
                                f"Значение должно быть положительным.")
                valid = False
        
        # Проверка диапазонов параметров
        range_params = [
            ('dynamics.damping.linear', self.dynamics['damping']['linear'], 0.0, 1.0),
            ('dynamics.damping.angular', self.dynamics['damping']['angular'], 0.0, 1.0),
            ('dynamics.integration.time_step', self.dynamics['integration']['time_step'], 0.0001, 0.1),
            ('placement.stabilization.energy_threshold', self.placement['stabilization']['energy_threshold'], 0.001, 10.0)
        ]
        
        for param_name, value, min_val, max_val in range_params:
            if not (min_val <= value <= max_val):
                self.logger.error(f"Некорректное значение параметра {param_name}: {value}. "
                                f"Допустимый диапазон: [{min_val}, {max_val}].")
                valid = False
        
        # Проверка режимов и перечислений
        valid_modes = ['sequential', 'parallel', 'hybrid']
        if self.placement['mode'] not in valid_modes:
            self.logger.error(f"Некорректный режим размещения: {self.placement['mode']}. "
                            f"Допустимые значения: {valid_modes}")
            valid = False
        
        valid_technologies = list(self.technological_constraints['min_gaps'].keys())
        if self.technological_constraints['cutting_technology'] not in valid_technologies:
            self.logger.error(f"Некорректная технология резки: {self.technological_constraints['cutting_technology']}. "
                            f"Допустимые значения: {valid_technologies}")
            valid = False
        
        if not valid:
            self.logger.error("Конфигурация содержит ошибки и не может быть использована")
        
        return valid
    
    def get_sheet_size_from_input(self, input_data: Union[str, Dict[str, Any]]) -> Tuple[float, float]:
        """
        Получение размеров листа из входных данных
        
        :param input_data: Имя файла или словарь с данными
        :return: Размеры листа (ширина, высота) в мм
        """
        try:
            if isinstance(input_data, str):
                # Загрузка данных из файла
                with open(input_data, 'r', encoding='utf-8') as f:
                    if input_data.endswith('.yaml') or input_data.endswith('.yml'):
                        data = yaml.safe_load(f)
                    elif input_data.endswith('.json'):
                        data = json.load(f)
                    else:
                        # Попытка загрузить как DXF/STEP через соответствующие модули
                        from my_io.dxf_import import import_dxf
                        from my_io.step_import import import_step
                        
                        if input_data.endswith('.dxf'):
                            from my_io.dxf_import import get_sheet_size_from_dxf
                            sheet_size = get_sheet_size_from_dxf(input_data)
                            if sheet_size:
                                return sheet_size
                            
                            return (2000.0, 1000.0)
                        elif input_data.endswith('.step'):
                            from my_io.step_import import import_step
                            shapes = import_step(input_data)
                            # Аналогичный расчет габаритов
                            min_x = min_y = float('inf')
                            max_x = max_y = float('-inf')
                            for shape in shapes:
                                bbox = shape.get_bounding_box()
                                min_x = min(min_x, bbox[0])
                                min_y = min(min_y, bbox[1])
                                max_x = max(max_x, bbox[2])
                                max_y = max(max_y, bbox[3])
                            return (max_x - min_x, max_y - min_y)
                        else:
                            raise ValueError(f"Неподдерживаемый формат файла: {input_data}")
            else:
                data = input_data
            
            # Поиск размеров листа в данных
            if 'sheet_size' in data:
                return tuple(data['sheet_size'])
            elif 'width' in data and 'height' in data:
                return (data['width'], data['height'])
            elif 'bounds' in data:
                bounds = data['bounds']
                return (bounds[2] - bounds[0], bounds[3] - bounds[1])
            
            # Попытка извлечь из метаданных DXF/STEP
            if 'metadata' in data and 'sheet_size' in data['metadata']:
                return tuple(data['metadata']['sheet_size'])
            
            # Значения по умолчанию
            self.logger.warning("Размеры листа не найдены в входных данных. Используются значения по умолчанию.")
            return (2000.0, 1000.0)  # мм (стандартный лист)
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении размеров листа из входных данных: {e}")
            return (2000.0, 1000.0)  # мм
    
    def __str__(self):
        """Строковое представление для отладки"""
        current_profile = self.get_current_profile()
        return (f"SystemConfig(profile='{current_profile}', "
                f"mode='{self.placement['mode']}', "
                f"technology='{self.technological_constraints['cutting_technology']}', "
                f"min_gap={self.geometry['collision_detection']['min_gap']} мм, "
                f"tolerance={self.geometry['polygonization']['tolerance']} мм)")


# Глобальный экземпляр конфигурации для удобного доступа
_config_instance = None

def get_config(config_file: Optional[str] = None) -> SystemConfig:
    """
    Получение глобального экземпляра конфигурации
    
    :param config_file: Путь к файлу конфигурации (загружается только при первом вызове)
    :return: Экземпляр SystemConfig
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = SystemConfig(config_file)
    
    return _config_instance


def load_profile(profile_name: str):
    """
    Загрузка профиля конфигурации
    
    :param profile_name: Имя профиля
    """
    config = get_config()
    config.apply_profile(profile_name)


def reset_config():
    """
    Сброс конфигурации к значениям по умолчанию
    """
    global _config_instance
    _config_instance = None
    return get_config()