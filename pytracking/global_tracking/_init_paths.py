import sys
import os
from vot_path import base_path
sys.path.insert(0, os.path.join(base_path, 'pytracking/global_tracking/_submodules/neuron'))
sys.path.insert(0, os.path.join(base_path, 'pytracking/global_tracking/_submodules/mmdetection'))
sys.path.insert(0, os.path.join(base_path, 'pytracking/global_tracking'))

from modules import *
from datasets import *
