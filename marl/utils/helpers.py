from itertools import count
import torch
import gymnasium as gym

from marl.utils.plotter import Plotter
from marl.utils.training import Training
from marl.model.dqn import DQN
from marl.utils.replay_memory import ReplayMemory
from marl.utils.constants import *
from marl.environment.reentrancy_env import ReentrancyEnv
from marl.agents.attacker import Attacker
from marl.agents.defender import Defender
from marl.utils.logger import logger


def initialize_environment_and_agents():
    """
    Initialize the environment and the agents.

    Returns:
        tuple: A tuple containing the environment, attacker and defender agents,
               and their respective policy networks.
    """
    # Initialize the custom environment
    env = ReentrancyEnv()

    # Determine the number of actions from the environment's action space
    n_actions = (
        env.action_space.n
        if isinstance(env.action_space, gym.spaces.Discrete)
        else None
    )

    # Assuming a binary state representation
    n_observations = 1

    # Initialize DQN models for both attacker and defender
    attacker_policy_net = DQN(n_observations, n_actions).to(DEVICE)
    defender_policy_net = DQN(n_observations, n_actions).to(DEVICE)

    # Create attacker and defender agents
    attacker = Attacker(attacker_policy_net, EPS_START, EPS_END, EPS_DECAY)
    defender = Defender(defender_policy_net, EPS_START, EPS_END, EPS_DECAY)

    return env, attacker, defender, attacker_policy_net, defender_policy_net


def initialize_training_memory_and_plotter(
    attacker_policy_net, defender_policy_net
):
    """
    Initialize the replay memory, training module, and plotter.

    Args:
        attacker_policy_net (DQN): The policy network for the attacker.
        defender_policy_net (DQN): The policy network for the defender.

    Returns:
        tuple: A tuple containing the replay memory, training module, and plotter.
    """
    # Initialize replay memory
    memory = ReplayMemory(10000)

    # Create optimizer for both attacker and defender policy networks
    optimizer = torch.optim.Adam(
        list(attacker_policy_net.parameters())
        + list(defender_policy_net.parameters()),
        lr=LR,
        amsgrad=True,
    )

    # Initialize the training module
    training = Training(
        attacker_policy_net,
        defender_policy_net,
        optimizer,
        memory,
        TAU,
        GAMMA,
        BATCH_SIZE,
    )

    # Initialize the plotter for visualizing training progress
    plotter = Plotter()

    return memory, training, plotter


def run_episode(
    env,
    attacker,
    defender,
    memory,
    training,
    plotter,
    episode,
    num_episodes,
    attacker_cumulative_reward,
    defender_cumulative_reward,
):
    """
    Run a single episode of the training loop.

    Args:
        env (gym.Env): The training environment.
        attacker (Attacker): The attacker agent.
        defender (Defender): The defender agent.
        memory (ReplayMemory): The replay memory.
        training (Training): The training module.
        plotter (Plotter): The plotter for visualizing results.
        episode (int): The current episode number.
        num_episodes (int): The total number of episodes.
        attacker_cumulative_reward (float): The cumulative reward of the attacker.
        defender_cumulative_reward (float): The cumulative reward of the defender.

    Returns:
        tuple: A tuple containing the updated cumulative rewards for both the attacker and the defender.
    """
    # Reset the environment and get the initial state
    state = torch.tensor(
        [env.reset()], device=DEVICE, dtype=torch.float32
    ).unsqueeze(0)

    logger.info(f"Episode {episode + 1}/{num_episodes}")

    for t in count():
        # Select actions for both attacker and defender
        attacker_action = attacker.select_action(state)
        defender_action = defender.select_action(state)

        # Execute actions in the environment and observe the next state and rewards
        next_state, (attacker_reward, defender_reward), done, _ = env.step(
            (attacker_action.item(), defender_action.item())
        )
        next_state = torch.tensor(
            [next_state], device=DEVICE, dtype=torch.float32
        )

        # Update cumulative rewards
        attacker_cumulative_reward += attacker_reward
        defender_cumulative_reward += defender_reward

        # Log information about the current step
        log_step_info(
            t,
            state,
            attacker_action,
            defender_action,
            attacker_reward,
            defender_reward,
            attacker_cumulative_reward,
            defender_cumulative_reward,
        )

        # Store the transition in replay memory
        memory.push(
            state,
            torch.tensor([[attacker_action, defender_action]], device=DEVICE),
            next_state,
            torch.tensor([attacker_reward, defender_reward], device=DEVICE),
        )

        # Update the current state
        state = next_state

        # Perform optimization step
        training.optimize_model()

        # Check if the episode is done
        if done:
            # Update the plotter with the latest rewards and plot the results
            plotter.append_rewards(
                attacker_cumulative_reward, defender_cumulative_reward
            )
            plotter.plot_rewards()
            break

    # Update the target network
    training.update_target_net()

    return attacker_cumulative_reward, defender_cumulative_reward


def log_step_info(
    t,
    state,
    attacker_action,
    defender_action,
    attacker_reward,
    defender_reward,
    attacker_cumulative_reward,
    defender_cumulative_reward,
):
    """
    Log information about the current step of the episode.

    Args:
        t (int): The current step number within the episode.
        state (torch.Tensor): The current state.
        attacker_action (int): The action taken by the attacker.
        defender_action (int): The action taken by the defender.
        attacker_reward (float): The reward received by the attacker.
        defender_reward (float): The reward received by the defender.
        attacker_cumulative_reward (float): The cumulative reward of the attacker.
        defender_cumulative_reward (float): The cumulative reward of the defender.
    """
    logger.info(f"Step {t + 1}:")
    logger.info(f"  Current State: {state.item()}")
    logger.info(
        f"  Attacker Action: {attacker_action.item()}, Reward: {attacker_reward}"
    )
    logger.info(
        f"  Defender Action: {defender_action.item()}, Reward: {defender_reward}"
    )
    logger.info(f"  Cumulative Attacker Reward: {attacker_cumulative_reward}")
    logger.info(f"  Cumulative Defender Reward: {defender_cumulative_reward}")
    logger.info("-" * 30)
