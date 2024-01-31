"""
Reentrancy Environment Module.

This module defines a custom environment compatible with the Gymnasium framework, simulating
a scenario where an attacker and a defender interact with smart contracts that may or may not
contain reentrancy vulnerabilities.

The environment is designed to represent a binary state space where each state corresponds to
whether a smart contract is deployed with a reentrancy vulnerability or not. Similarly, the
action space is binary, representing the decision to deploy a contract with or without a
vulnerability.

Classes:
    ReentrancyEnv: A Gymnasium environment simulating interactions with smart contracts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from marl.utils.logger import logger


class ReentrancyEnv(gym.Env):
    """
    A Gymnasium environment for simulating interactions with smart contracts regarding
    reentrancy vulnerabilities.

    The environment represents a simple model where an attacker deploys contracts and
    a defender attempts to identify vulnerabilities. The state and action spaces are
    binary, representing the presence or absence of a vulnerability.

    Attributes:
        action_space (gym.spaces.Discrete): The action space, binary in this case.
        observation_space (gym.spaces.Discrete): The state space, also binary.
        state (int): The current state of the environment.
    """

    def __init__(self):
        """
        Initializes the ReentrancyEnv environment.

        Sets up the action and observation spaces to be binary, representing the decisions
        to deploy contracts with or without vulnerabilities and the state of the deployed
        contract, respectively.
        """
        # Binary actions: deploy with/without vulnerability
        self.action_space = spaces.Discrete(2)
        # Binary state: contract with/without vulnerability
        self.observation_space = spaces.Discrete(2)

    def reset(self):
        """
        Resets the environment to a new initial state.

        Returns:
            int: The initial state of the environment after reset.
        """
        # Randomly initialize the state
        self.state = np.random.randint(0, 2)
        return self.state

    def step(self, actions):
        """
        Executes a step in the environment given an action.

        Args:
            actions (tuple): A pair of actions (attacker_action, defender_action).

        Returns:
            tuple: A tuple containing the new state, the reward tuple (attacker_reward, defender_reward),
                   a boolean indicating if the episode is done, and an empty dictionary.
        """
        attacker_action, defender_action = actions
        done = False

        # Update state based on attacker's action
        self.state = attacker_action

        # Reward mechanism
        if defender_action == self.state:
            defender_reward = 1
            attacker_reward = -1 if attacker_action == 1 else 0
        else:
            defender_reward = -1
            attacker_reward = 1 if attacker_action == 1 else 0

        done = True
        return self.state, (attacker_reward, defender_reward), done, {}

    def render(self, mode="console"):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment. Currently, only 'console' mode is supported.
        """
        if mode == "console":
            if self.state == 1:
                logger.info("Contract with Reentrancy Vulnerability Deployed")
            else:
                logger.info("Safe Contract Deployed")
        else:
            raise NotImplementedError(
                "Only 'console' mode is supported for rendering."
            )
