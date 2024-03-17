"""
Reentrancy Environment Module.

This module defines a custom environment compatible with the Gymnasium framework, simulating
a scenario where an deployer and a detector interact with smart contracts that may or may not
contain reentrancy vulnerabilities.

The environment uses ABIs of smart contracts to simulate the behavior of deploying contracts
with or without vulnerabilities. The state and action spaces are binary, representing the 
decision to deploy a contract with or without a vulnerability, and the detector's attempt to 
identify these vulnerabilities.

Classes:
    ReentrancyEnv: A Gymnasium environment simulating interactions with smart contracts.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from marl.utils.logger import logger
from marl.environment.utils.contract_analysis import *


class ReentrancyEnv(gym.Env):
    """
    A Gymnasium environment for simulating interactions with smart contracts regarding
    reentrancy vulnerabilities.

    The environment represents a simple model where an deployer deploys contracts and
    a detector attempts to identify vulnerabilities. The state and action spaces are
    binary, representing the presence or absence of a vulnerability.

    Attributes:
        action_space (gym.spaces.Discrete): The action space, binary in this case.
        observation_space (gym.spaces.Discrete): The state space, also binary.
        state (int): The current state of the environment.
        secure_contract (dict): The ABI of a malicious contract.
        secure_contract (dict): The ABI of a non-malicious contract.
    """

    def __init__(self):
        """
        Initializes the ReentrancyEnv environment.

        Sets up the action and observation spaces to be binary, representing the decisions
        to deploy contracts with or without vulnerabilities and the state of the deployed
        contract, respectively.
        """
        super().__init__()
        # Action space binary; 0 for choosing contract_1, 1 for choosing contract_2
        self.action_space = spaces.Discrete(2)

        # Observation space now includes a feature for vulnerability analysis
        self.observation_space = spaces.Dict(
            {
                "externalCalls": spaces.Discrete(2),
                "stateUpdates": spaces.Discrete(2),
                "functionTypes": spaces.MultiBinary(
                    4
                ),  # Example: [public, external, private, internal]
                "pattern_presence": spaces.Dict(
                    {
                        "call_value": spaces.Discrete(2),
                        "delegatecall": spaces.Discrete(2),
                        "selfdestruct": spaces.Discrete(2),
                    }
                ),
            }
        )

        # Load and analyze contracts
        self.contracts = [
            load_contract("contract_1.sol"),
            load_contract("contract_2.sol"),
        ]

    def reset(self):
        """
        Resets the environment to a new initial state.

        Returns:
            int: The initial state of the environment after reset.
        """
        """
        Resets the environment to a new initial state, randomly choosing a contract.
        """
        # Randomly choose between contract_1 and contract_2
        self.state = {
            "externalCalls": np.random.randint(0, 2),
            "stateUpdates": np.random.randint(0, 2),
            "functionTypes": np.random.randint(
                0, 2, size=4
            ).tolist(),  # Assuming a list is suitable for your setup
            "pattern_presence": {
                "call_value": np.random.randint(0, 2),
                "delegatecall": np.random.randint(0, 2),
                "selfdestruct": np.random.randint(0, 2),
            },
        }
        return self.state

    def step(self, actions):
        """
        Executes a step in the environment given an action.

        Args:
            actions (tuple): A pair of actions (deployer_action, detector_action).

        Returns:
            tuple: A tuple containing the new state, the reward tuple (deployer_reward, detector_reward),
                   a boolean indicating if the episode is done, and an empty dictionary.
        """
        deployer_action, detector_action = actions

        # The state directly reflects the contract's vulnerability (0 for non-vulnerable, 1 for vulnerable)
        contract_features = self.contracts[deployer_action]

        # Use deployer_action to select the contract
        is_vulnerable = (
            1
            if "reentrancy" in contract_features.get("vulnerabilities", [])
            else 0
        )

        # Initialize rewards
        deployer_reward = 0
        detector_reward = 0

        # Sequence of detection flow
        # Logic for detector reward
        if is_vulnerable:
            if detector_action == 1:
                # Detector correctly identifies a vulnerable contract
                detector_reward = 1
            else:
                # Detector fails to identify a vulnerable contract
                detector_reward = -1
                deployer_reward = 1
        else:
            if detector_action == 0:
                # Detector correctly identifies a non-vulnerable contract
                detector_reward = 1
            else:
                # Detector incorrectly identifies a non-vulnerable contract as vulnerable
                detector_reward = -1
                deployer_reward = 1

        # Assuming each step ends the episode for simplicity
        done = True
        return self.state, (deployer_reward, detector_reward), done, {}

    def render(self, mode="console"):
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode in which to render the environment. Currently, only 'console' mode is supported.
        """
        if mode == "console":
            contract_name = (
                "contract_1.sol" if self.state == 0 else "contract_2.sol"
            )
            logger.info(f"{contract_name} Deployed")
        else:
            raise NotImplementedError(
                "Only 'console' mode is supported for rendering."
            )
