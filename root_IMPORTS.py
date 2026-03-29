"""
Общий файл импортов
"""

#region IMPORTS
import os 
import re
import sys
import json 
import time
import math
import yaml
import ezdxf
import numpy
import rtree
import random
import argparse
import logging 
import tempfile
import warnings
import threading
import traceback

import typing
from typing import List, Dict, Tuple, Optional, Any

import pathlib
from pathlib import Path as pl_Path 

#region scipy 
import scipy
import scipy.interpolate
import scipy.spatial
#endregion scipy

#region shapely 
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
#endregion shapely

#region matplotlib
import matplotlib
import matplotlib.backends.backend_qt5agg
import matplotlib.collections
import matplotlib.colors
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot
#endregion matplotlib

#region OCC
import OCC.Core.BRep
import OCC.Core.BRepAdaptor
import OCC.Core.BRepBuilderAPI
import OCC.Core.BRepTools
import OCC.Core.GCPnts
import OCC.Core.Geom
import OCC.Core.IFSelect
import OCC.Core.Interface
import OCC.Core.Precision
import OCC.Core.STEPControl
import OCC.Core.Standard
import OCC.Core.TopAbs
import OCC.Core.TopExp
import OCC.Core.TopoDS
import OCC.Core.gp
#endregion OCC

#region PYQT
import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets
#endregion PYQT

import xml.etree.ElementTree
import concurrent.futures
#endregion IMPORTS

