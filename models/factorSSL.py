from .critic_objectives import*

from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )



