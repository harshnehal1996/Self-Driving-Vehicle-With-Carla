import torch
import glob
import sys
from paths import ProjectPaths
sys.path.append(glob.glob(ProjectPaths.carla_pylibs)
                [0])  # import carla python library

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
