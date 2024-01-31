from collections import namedtuple
import torch
from torch import optim

# Constants
BATCH_SIZE: int = 128
GAMMA: float = 0.99
EPS_START: float = 0.9
EPS_END: float = 0.05
EPS_DECAY: int = 1000
TAU: float = 0.005
LR: float = 1e-4

# Transition tuple
TRANSITION = namedtuple(
    "TRANSITION", ("state", "action", "next_state", "reward")
)

# GPU usage configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
