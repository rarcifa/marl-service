"""
Training Module.

This module provides a Training class for managing the training process of a reinforcement learning agent
using a Deep Q-Network (DQN).

The Training class is responsible for optimizing the DQN model, updating the target network, and handling
the sampling of experiences from the replay memory. This class plays a pivotal role in updating the agent's
knowledge and improving its performance over time through learning.

Classes:
    Training: Manages the training process of a DQN-based reinforcement learning agent.
"""

import torch.nn as nn

from marl.model.dqn import DQN
from marl.utils.replay_memory import ReplayMemory
from marl.utils.constants import *
from marl.utils.logger import logger


class Training:
    """
    Manages the training process of a reinforcement learning agent using a Deep Q-Network (DQN).

    The Training class is responsible for the optimization of the policy network, updates to the target network,
    and managing the replay memory. It uses experiences from the memory to update the agent's policy.

    Attributes:
        policy_net (DQN): The DQN model used as the policy network.
        target_net (DQN): A separate DQN model used as the target network.
        optimizer (torch.optim.Optimizer): Optimizer for training the policy network.
        memory (ReplayMemory): Memory buffer storing experiences for replay.
        tau (float): Rate at which the target network is updated (soft update).
        gamma (float): Discount factor for future rewards.
        batch_size (int): Size of batches sampled from the memory for training.
    """

    def __init__(
        self,
        policy_net: DQN,
        target_net: DQN,
        optimizer: torch.optim.Optimizer,
        memory: ReplayMemory,
        tau: float,
        gamma: float,
        batch_size: int,
    ):
        """
        Initializes the Training class with policy and target networks, optimizer, and training parameters.

        Args:
            policy_net (DQN): Policy network for the agent.
            target_net (DQN): Target network for stability in Q-learning.
            optimizer (torch.optim.Optimizer): Optimizer for the policy network.
            memory (ReplayMemory): Experience replay memory.
            tau (float): Rate of updating the target network.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Batch size for sampling from memory.
        """
        self.policy_net: DQN = policy_net
        self.target_net: DQN = target_net
        self.optimizer: optim.Optimizer = optimizer
        self.memory: ReplayMemory = memory
        self.tau: float = tau
        self.gamma: float = gamma
        self.batch_size: int = batch_size

    def optimize_model(self) -> None:
        """
        Optimizes the policy network using experiences sampled from the replay memory.

        This method samples a batch of transitions from the replay memory and uses them to update
        the policy network. The update involves calculating the loss based on the current state and
        action values and the expected Q values for the next states. The method uses the Huber loss
        for stability.

        After calculating the loss, the method performs backpropagation and updates the weights of
        the policy network using the optimizer. This optimization step is crucial for the learning
        process of the agent.
        """
        # Check if enough samples are available in memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)

        # Unzip transitions to a batch
        batch = TRANSITION(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (non-final states are states where next_state is not None)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=DEVICE,
            dtype=torch.bool,
        )

        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(DEVICE)

        # Correct the handling here by ensuring we reshape or adjust non_final_next_states correctly
        # The shape should be [number_of_non_final_states, 9], without an extra dimension
        if non_final_next_states.nelement() != 0:
            if (
                non_final_next_states.dim() == 3
                and non_final_next_states.shape[1] == 1
            ):
                # Remove the middle dimension if it exists and is equal to 1
                non_final_next_states = non_final_next_states.squeeze(1)
            elif non_final_next_states.shape[-1] != 9:
                # Log an error if the last dimension is not 9, indicating a problem with state processing
                logger.error(
                    f"Incorrect non_final_next_states shape: {non_final_next_states.shape}"
                )
                # Consider adding corrective action or more detailed logging here

        # Reshape non_final_next_states to the correct shape [batch_size, n_observations]
        if (
            non_final_next_states.nelement() != 0
        ):  # Check if the tensor is not empty
            non_final_next_states = non_final_next_states.view(-1, 1)

        # Concatenate the state, action, and reward batches
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(
            [reward[1].unsqueeze(0) for reward in batch.reward]
        )

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # Compute V(s_{t+1}) for all next states, using expected values based on the older target_net
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        with torch.no_grad():
            if (
                non_final_next_states.nelement() != 0
            ):  # Check if the tensor is not empty

                batch_size = non_final_next_states.size(0) // 9
                non_final_next_states = non_final_next_states.view(
                    batch_size, 9
                )
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).max(1).values
                )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch

        # Compute Huber loss between expected Q values and the model's Q values
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self) -> None:
        """
        Updates the weights of the target network.

        This method performs a soft update of the target network's weights using the weights from
        the policy network. The update is controlled by the tau parameter, which determines the
        rate at which the target network is updated.

        The soft update ensures gradual changes to the target network, contributing to the stability
        of the learning process.
        """

        # Retrieve the current state dictionaries of the target and policy networks
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # Loop through the parameters in the policy network
        for key in policy_net_state_dict:
            # Perform the soft update: a weighted sum of the target and policy network parameters
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)

        # Load the updated state dictionary back into the target network
        self.target_net.load_state_dict(target_net_state_dict)
