"""
Общие импорты для серверной части приложения ИАГИ
Централизованное управление зависимостями
"""

# Стандартные библиотеки
import sys
import os
import time
import json
import logging
import threading
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
import traceback
import uuid
import hashlib

# Сторонние библиотеки
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Локальные модули - Core
from core.geometry import PolygonShape
from core.agent import IAGIAgent as Agent
from core.dynamics import GravitationalDynamics as DynamicsEngine
from core.constraints import ConstraintManager
from core.collision import CollisionDetector
from core.composite import CompositeShape
from core.optimizer import PackingOptimizer
from core.remnant_manager import RemnantManager
from core.sheet_batch import SheetBatchManager as SheetBatch

# Локальные модули - Algorithms
from algorithms.sequential import sequential_placement
from algorithms.parallel import parallel_placement
from algorithms.hybrid import hybrid_optimization_factory as hybrid_placement
from algorithms.multi_start import multi_start_optimization
from algorithms.batch_coordinator import BatchNestingCoordinator as BatchCoordinator
from algorithms.batch_distribution import BatchDistributionOptimizer
from algorithms.priority_manager import PriorityManager
from algorithms.stabilization import check_energy_stagnation, apply_stabilization

StabilizationController = None  # Функции вместо класса

# Локальные модули - IO
from my_io.dxf_import import import_dxf
from my_io.step_import import import_step
from my_io.exporter import export_results

# Локальные модули - Config
from config.settings import get_config, load_profile, save_profile


__all__ = [
    # Стандартные
    'sys', 'os', 'time', 'json', 'logging', 'threading', 
    'tempfile', 'shutil', 'Path', 'List', 'Dict', 'Tuple', 
    'Any', 'Optional', 'Callable', 'datetime', 'dataclass', 
    'field', 'contextmanager', 'traceback', 'uuid', 'hashlib',
    
    # Сторонние
    'np', 'FastAPI', 'UploadFile', 'File', 'Form', 'HTTPException',
    'BackgroundTasks', 'JSONResponse', 'FileResponse', 'CORSMiddleware',
    'uvicorn',
    
    # Core
    'PolygonShape', 'Agent', 'DynamicsEngine', 'ConstraintManager',
    'CollisionDetector', 'CompositeShape', 'PackingOptimizer',
    'RemnantManager', 'SheetBatch',
    
    # Algorithms
    'sequential_placement', 'parallel_placement', 'hybrid_placement',
    'multi_start_optimization', 'BatchCoordinator', 
    'BatchDistributionOptimizer', 'PriorityManager', 'StabilizationController',
    
    # IO
    'import_dxf', 'import_step', 'export_results',
    
    # Config
    'get_config', 'load_profile', 'save_profile'
]
