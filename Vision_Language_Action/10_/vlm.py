'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2025-03-30 14:18:55
LastEditTime: 2025-03-30 14:19:46
'''


from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.lora import get_layer
from src.model.paligemma.modules import (
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
)