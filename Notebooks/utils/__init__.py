"""
UtilitiesPackage for Federated Learning IoT Attack Detection

This package provides data processing, model utilities, FL training logic,
feature selection (GSA), and advanced aggregation (FedMade).
"""

# Import commonly used functions for easy access
from . import data_utils
from . import model_utils_pytorch
from . import fl_utils_pytorch
from . import gsa_algorithm
from . import fedmade_aggregation
from . import feature_utils
from . import includes
from . import fl_utils_pytorch

__all__ = [
    'data_utils',
    'model_utils_pytorch',
    'fl_utils_pytorch',
    'gsa_algorithm',
    'fedmade_aggregation',
    'feature_utils',
    'includes',
    'fl_utils_pytorch'
]
__version__ = '2.0.0'  # Updated for GSA + FedMade
