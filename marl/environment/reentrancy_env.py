"""
Reentrancy Environment Module.

This module defines a custom environment compatible with the Gymnasium framework, simulating
a scenario where an attacker and a defender interact with smart contracts that may or may not
contain reentrancy vulnerabilities.

The environment uses ABIs of smart contracts to simulate the behavior of deploying contracts
with or without vulnerabilities. The state and action spaces are binary, representing the 
decision to deploy a contract with or without a vulnerability, and the defender's attempt to 
identify these vulnerabilities.

Classes:
    ReentrancyEnv: A Gymnasium environment simulating interactions with smart contracts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from marl.utils.logger import logger
from marl.environment.utils.helpers import *


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
        malicious_contract_abi (dict): The ABI of a malicious contract.
        good_contract_abi (dict): The ABI of a non-malicious contract.
    """

    def __init__(self):
        """
        Initializes the ReentrancyEnv environment.

        Sets up the action and observation spaces to be binary, representing the decisions
        to deploy contracts with or without vulnerabilities and the state of the deployed
        contract, respectively.
        """
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.malicious_contract_abi = load_abi("malicious_contract_abi.json")
        self.good_contract_abi = load_abi("normal_contract_abi.json")

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

        # Determine if the ABI is malicious based on the attacker's action
        contract_abi = (
            self.malicious_contract_abi
            if attacker_action == 1
            else self.good_contract_abi
        )

        is_malicious = is_malicious_abi(contract_abi)
        self.state = 1 if is_malicious else 0

        # Assign rewards based on the specified guidelines
        # Attacker deploys a non-malicious contract (action 0)
        if attacker_action == 0:
            attacker_reward = 0
            defender_reward = 1 if defender_action == 0 else -1

        # Attacker deploys a malicious contract (action 1)
        else:
            attacker_reward = 1 if defender_action == 0 else -1
            defender_reward = -1 if defender_action == 0 else 1

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
