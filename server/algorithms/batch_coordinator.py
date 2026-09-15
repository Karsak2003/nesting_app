
import time
import numpy as np
from typing import List, Dict,Tuple, Optional, Any
from core.sheet_batch import SheetBatchManager, Sheet
from algorithms.batch_distribution import BatchDistributionOptimizer
from core.optimizer import PackingOptimizer
from core.constraints import ConstraintManager
import logging

"""
Модуль координации раскроя партии листовых материалов
Согласно разделу 3.2.5, глобальная координация критична для достижения оптимальной плотности упаковки
"""

logger = logging.getLogger(__name__)

class BatchNestingCoordinator:
    """
    Координатор раскроя партии листовых материалов
    
    Обеспечивает:
    - Последовательную обработку листов с учетом приоритетов
    - Оптимизацию общего использования материала
    - Минимизацию переходов между листами
    - Учет остатков и повторное использование
    """
    
    def __init__(self,
                 sheet_batch: SheetBatchManager,
                 constraint_manager: ConstraintManager,
                 distribution_method: str = 'hybrid_ga',
                 time_limit: float = 600.0):
        """
        Инициализация координатора раскроя партии
        
        :param sheet_batch: Менеджер партии листов
        :param constraint_manager: Менеджер технологических ограничений
        :param distribution_method: Метод распределения ('first_fit', 'best_fit', 'hybrid_ga', 'hybrid_pso')
        :param time_limit: Лимит времени на полную оптимизацию партии
        """
        self.sheet_batch = sheet_batch
        self.constraint_manager = constraint_manager
        self.distribution_method = distribution_method
        self.time_limit = time_limit
        
        # Инициализация оптимизатора распределения
        self.distribution_optimizer = BatchDistributionOptimizer(
            sheet_batch=sheet_batch,
            constraint_manager=constraint_manager,
            optimization_method=distribution_method,
            time_limit=time_limit / 2  # Половина времени для распределения
        )
        
        # Статистика координации
        self.coordination_stats = {
            'total_sheets_processed': 0,
            'total_parts_processed': 0,
            'total_utilization': 0.0,
            'coordination_time': 0.0,
            'sheets_by_priority': []
        }
    
    def coordinate_batch_nesting(self,
                                parts: List[Any],
                                priorities: Optional[List[int]] = None,
                                orientation_constraints: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Координация раскроя партии листовых материалов
        
        :param parts: Список деталей для раскроя
        :param priorities: Приоритеты деталей
        :param orientation_constraints: Ограничения на ориентацию
        :return: Словарь с результатами координации
        """
        start_time = time.time()
        logger.info(f"Начало координации раскроя партии: {len(parts)} деталей, "
                   f"{len(self.sheet_batch)} листов")
        
        # Шаг 1: Распределение деталей по листам
        logger.info("Шаг 1: Распределение деталей по листам...")
        distribution_result = self.distribution_optimizer.distribute_parts(
            parts=parts,
            priorities=priorities,
            orientation_constraints=orientation_constraints
        )
        
        distribution = distribution_result['distribution']
        
        # Шаг 2: Сортировка листов по приоритету для обработки
        logger.info("Шаг 2: Сортировка листов по приоритету...")
        sorted_sheet_ids = sorted(
            distribution.keys(),
            key=lambda sid: distribution[sid]['sheet'].priority
        )
        
        # Шаг 3: Последовательная обработка каждого листа
        logger.info("Шаг 3: Последовательная обработка листов...")
        sheet_results = {}
        
        for sheet_id in sorted_sheet_ids:
            sheet_start_time = time.time()
            
            sheet_data = distribution[sheet_id]
            sheet = sheet_data['sheet']
            parts_for_sheet = [item['part'] for item in sheet_data['parts']]
            priorities_for_sheet = [item['priority'] for item in sheet_data['parts']]
            orientation_constraints_for_sheet = [item['orientation_constraint'] for item in sheet_data['parts']]
            
            logger.info(f"Обработка листа {sheet_id} (приоритет {sheet.priority})...")
            
            # Создание оптимизатора для листа
            optimizer = PackingOptimizer({
                'sheet_size': sheet.get_sheet_size(),
                'min_gap': sheet.min_gap,
                'time_limit': 120.0,  # 2 минуты на лист
                'use_sequential': True,
                'stabilization_enabled': True
            })
            
            # Добавление дефектных зон
            for defect in sheet.defects:
                optimizer.add_defect_zone(defect)
            
            # Запуск оптимизации раскроя
            try:
                result_agents = optimizer.optimize(
                    shapes=parts_for_sheet,
                    priorities=priorities_for_sheet,
                    orientation_constraints=orientation_constraints_for_sheet
                )
                
                # Расчет утилизации
                total_area = sum(agent.shape.area for agent in result_agents)
                sheet_area = sheet.width * sheet.height
                utilization = (total_area / sheet_area) * 100.0
                
                # Сохранение результатов
                sheet_results[sheet_id] = {
                    'sheet': sheet,
                    'agents': result_agents,
                    'utilization': utilization,
                    'processing_time': time.time() - sheet_start_time,
                    'num_parts': len(parts_for_sheet)
                }
                
                # Пометка листа как использованного
                self.sheet_batch.mark_sheet_as_used(
                    sheet_id=sheet_id,
                    utilization=utilization,
                    placed_parts=result_agents
                )
                
                logger.info(f"Лист {sheet_id} обработан: утилизация {utilization:.2f}%, "
                           f"время {sheet_results[sheet_id]['processing_time']:.2f}с")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке листа {sheet_id}: {e}")
                sheet_results[sheet_id] = {
                    'sheet': sheet,
                    'agents': [],
                    'utilization': 0.0,
                    'processing_time': time.time() - sheet_start_time,
                    'num_parts': 0,
                    'error': str(e)
                }
        
        # Шаг 4: Расчет итоговой статистики
        coordination_time = time.time() - start_time
        
        total_sheets = len(sheet_results)
        total_parts = sum(result['num_parts'] for result in sheet_results.values())
        total_utilization = sum(result['utilization'] for result in sheet_results.values())
        avg_utilization = total_utilization / total_sheets if total_sheets > 0 else 0.0
        
        self.coordination_stats = {
            'total_sheets_processed': total_sheets,
            'total_parts_processed': total_parts,
            'total_utilization': total_utilization,
            'avg_utilization': avg_utilization,
            'coordination_time': coordination_time,
            'sheets_by_priority': sorted_sheet_ids,
            'sheet_results': sheet_results
        }
        
        logger.info(f"Координация раскроя партии завершена за {coordination_time:.2f} секунд")
        logger.info(f"Обработано листов: {total_sheets}, деталей: {total_parts}")
        logger.info(f"Средняя утилизация: {avg_utilization:.2f}%")
        
        return {
            'coordination_stats': self.coordination_stats,
            'sheet_results': sheet_results,
            'distribution': distribution,
            'total_time': coordination_time
        }
    
    def optimize_batch_sequence(self, distribution: Dict[str, Any]) -> List[str]:
        """
        Оптимизация последовательности обработки листов
        :param distribution: Распределение деталей по листам
        :return: Оптимизированная последовательность листов
        """
        # Базовая оптимизация: сортировка по приоритету и утилизации
        sheet_ids = list(distribution.keys())
        
        # Сортировка по приоритету листа, затем по ожидаемой утилизации
        sorted_sheets = sorted(
            sheet_ids,
            key=lambda sid: (
                distribution[sid]['sheet'].priority,
                -distribution[sid].get('utilization', 0.0)
            )
        )
        
        return sorted_sheets
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """
        Получение сводки по обработке партии
        
        :return: Словарь со сводкой
        """
        if not self.coordination_stats['sheet_results']:
            return {
                'status': 'not_processed',
                'message': 'Партия еще не обработана'
            }
        
        sheet_results = self.coordination_stats['sheet_results']
        
        # Статистика по листам
        utilizations = [result['utilization'] for result in sheet_results.values()]
        processing_times = [result['processing_time'] for result in sheet_results.values()]
        
        summary = {
            'status': 'processed',
            'total_sheets': self.coordination_stats['total_sheets_processed'],
            'total_parts': self.coordination_stats['total_parts_processed'],
            'avg_utilization': self.coordination_stats['avg_utilization'],
            'min_utilization': min(utilizations) if utilizations else 0.0,
            'max_utilization': max(utilizations) if utilizations else 0.0,
            'std_utilization': np.std(utilizations) if utilizations else 0.0,
            'total_processing_time': sum(processing_times),
            'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'sheets_by_utilization': sorted(
                [(sid, result['utilization']) for sid, result in sheet_results.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            'coordination_time': self.coordination_stats['coordination_time']
        }
        
        return summary
    
    def export_batch_results(self, output_dir: str = "results/batch"):
        """
        Экспорт результатов обработки партии
        
        :param output_dir: Директория для сохранения результатов
        """
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Экспорт сводки
        summary = self.get_batch_summary()
        summary_file = os.path.join(output_dir, "batch_summary.json")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Сводка по партии сохранена в {summary_file}")
        
        # Экспорт результатов по каждому листу
        for sheet_id, result in self.coordination_stats['sheet_results'].items():
            sheet_dir = os.path.join(output_dir, f"sheet_{sheet_id}")
            os.makedirs(sheet_dir, exist_ok=True)
            
            # Сохранение данных о листе
            sheet_data = {
                'sheet_id': sheet_id,
                'size': result['sheet'].get_sheet_size(),
                'material_type': result['sheet'].material_type,
                'utilization': result['utilization'],
                'num_parts': result['num_parts'],
                'processing_time': result['processing_time']
            }
            
            sheet_file = os.path.join(sheet_dir, "sheet_data.json")
            with open(sheet_file, 'w', encoding='utf-8') as f:
                json.dump(sheet_data, f, indent=2, ensure_ascii=False)
            
            # Экспорт раскроя в форматах DXF, SVG, JSON
            from my_io.exporter import export_results
            
            placements = []
            for agent in result['agents']:
                placements.append({
                    'shape': agent.shape,
                    'position': agent.position,
                    'angle': agent.angle,
                    'name': getattr(agent.shape, 'name', f"part_{len(placements)+1}"),
                    'priority': agent.priority
                })
            
            export_results(
                placements=placements,
                output_dir=sheet_dir,
                sheet_size=result['sheet'].get_sheet_size(),
                min_gap=result['sheet'].min_gap,
                defect_zones=result['sheet'].defects,
                formats=['dxf', 'svg', 'json']
            )
        
        logger.info(f"Результаты по всем листам сохранены в {output_dir}")