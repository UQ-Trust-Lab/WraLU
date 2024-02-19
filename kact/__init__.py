import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, '../../ELINA/python_interface/')
sys.path.insert(0, '../../ELINA/python_interface/tests')

from .group import *
from .krelu import *
