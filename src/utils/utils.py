import os
import random
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch


def set_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_useful_info() -> None:
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), 'src'),
        os.path.join(hydra.utils.get_original_cwd(), f'{os.getcwd()}/code/src'),
    )
    shutil.copy2(
        os.path.join(hydra.utils.get_original_cwd(), 'train.py'),
        os.path.join(hydra.utils.get_original_cwd(), os.getcwd(), 'code'),
    )


def check_dir(path):
    Path(path).mkdir(exist_ok=True, parents=True)