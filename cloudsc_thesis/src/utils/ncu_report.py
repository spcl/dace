import os
import sys
cuda_root = os.getenv('CUDA_ROOT')
for file in os.listdir(cuda_root):
    if os.path.isdir(os.path.join(cuda_root, file)) and file.startswith('nsight-compute'):
        nsight_folder = file
        break
sys.path.insert(0, os.path.join(cuda_root, nsight_folder, "extras", "python"))
from ncu_report import IAction, IMetric, load_report
