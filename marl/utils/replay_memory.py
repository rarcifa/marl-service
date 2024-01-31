"""
Replay Memory Module.

Module containing utility structures and classes used in reinforcement learning.

It includes a namedtuple for representing transitions in the environment and a class 
for storing these transitions in a replay memory for later retrieval during training.

Classes:
    ReplayMemory: A simple storage class for transitions observed during training.
"""
from collections import namedtuple, deque
from typing import Any, Deque, List
import random

from marl.utils.constants import *


class ReplayMemory:
    """
    Simple storage for transitions observed during training.

    This class provides a memory buffer that stores transitions up to a fixed maximum size.
    Transitions are stored in a deque, which automatically removes the oldest transitions
    once the capacity is exceeded.

    Attributes:
        memory (Deque[Transition]): A deque object to store transitions with a fixed maximum size.
    """

    def __init__(self, capacity: int):
        """
        Initializes the ReplayMemory with a given capacity.

        Args:
            capacity (int): The maximum number of items the memory can hold.
        """
        # Initialize the deque with the given capacity
        self.memory: Deque[TRANSITION] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        """
        Saves a transition to the memory.

        Args:
            *args: The transition data to be stored.
        """
        # Create and add the Transition to the deque
        self.memory.append(TRANSITION(*args))

    def sample(self, batch_size: int) -> List[Any]:
        """
        Randomly samples a batch of transitions from the memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            List[Any]: A list of sampled transitions.
        """
        # Randomly sample a batch of transitions
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of the memory.

        Returns:
            int: The number of items in the memory.
        """
        # Return the number of items in the deque
        return len(self.memory)
