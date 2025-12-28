"""
Utilities Package for Federated Learning IoT Attack Detection

This package contains utility modules for:
- data_utils: Data loading, preprocessing, and partitioning
- model_utils: Model architecture and compilation
- fl_utils: Federated learning client/server logic

Author: Nguyen Duc Thang
Project: IoT Network Attack Detection using Federated Learning
"""

from . import data_utils
from . import model_utils
from . import fl_utils

__all__ = ['data_utils', 'model_utils', 'fl_utils']
__version__ = '1.0.0'
