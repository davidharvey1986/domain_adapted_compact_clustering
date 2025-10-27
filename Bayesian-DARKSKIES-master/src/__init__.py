"""
Creates the logger
"""
import os
import re
import sys
import inspect
import logging
import warnings


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

# If using IPython, deprecation warnings are ignored as it causes anomalous warnings from several
# packages, so this re-enables it for all packages except problematic ones
regex = re.compile(r'^[a-zA-Z][a-zA-Z-]*')
warnings.filterwarnings('default', category=DeprecationWarning, module='src')
warnings.filterwarnings('default', category=PendingDeprecationWarning, module='src')

if os.path.exists('../requirements.txt'):
    with open('../requirements.txt', 'r', encoding='utf-8') as file:
        for package in file:
            if regex.match(package):
                warnings.filterwarnings(
                    'default',
                    category=DeprecationWarning,
                    module=regex.match(package)[0],
                )
                warnings.filterwarnings(
                    'default',
                    category=PendingDeprecationWarning,
                    module=regex.match(package)[0],
                )

try:
    import torch
    from src.utils import clustering, models
    from netloader.utils.utils import safe_globals

    # Adds PyTorch BaseNetwork classes to list of safe PyTorch classes when loading saved
    # networks
    safe_globals(__name__, [clustering, models])
except ModuleNotFoundError:
    pass
except ImportError:
    pass
