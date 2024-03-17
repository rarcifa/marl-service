from itertools import count
import torch
import gymnasium as gym

from marl.utils.plotter import Plotter
from marl.utils.training import Training
from marl.model.dqn import DQN
from marl.utils.replay_memory import ReplayMemory
from marl.utils.constants import *
from marl.environment.reentrancy_env import ReentrancyEnv
from marl.agents.deployer import Deployer
from marl.agents.detector import Detector
from marl.utils.logger import logger


def initialize_environment_and_agents():
    """
    Initialize the environment and the agents.

    Returns:
        tuple: A tuple containing the environment, deployer and detector agents,
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

    # Initialize DQN models for both deployer and detector
    deployer_policy_net = DQN(n_observations, n_actions).to(DEVICE)
    detector_policy_net = DQN(n_observations, n_actions).to(DEVICE)

    # Create deployer and detector agents
    deployer = Deployer(deployer_policy_net, EPS_START, EPS_END, EPS_DECAY)
    detector = Detector(detector_policy_net, EPS_START, EPS_END, EPS_DECAY)

    return env, deployer, detector, deployer_policy_net, detector_policy_net


def initialize_training_memory_and_plotter(
    deployer_policy_net, detector_policy_net
):
    """
    Initialize the replay memory, training module, and plotter.

    Args:
        deployer_policy_net (DQN): The policy network for the deployer.
        detector_policy_net (DQN): The policy network for the detector.

    Returns:
        tuple: A tuple containing the replay memory, training module, and plotter.
    """
    # Initialize replay memory
    memory = ReplayMemory(10000)

    # Create optimizer for both deployer and detector policy networks
    optimizer = torch.optim.Adam(
        list(deployer_policy_net.parameters())
        + list(detector_policy_net.parameters()),
        lr=LR,
        amsgrad=True,
    )

    # Initialize the training module
    training = Training(
        deployer_policy_net,
        detector_policy_net,
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
    deployer,
    detector,
    memory,
    training,
    plotter,
    episode,
    num_episodes,
    deployer_cumulative_reward,
    detector_cumulative_reward,
):
    """
    Run a single episode of the training loop.

    Args:
        env (gym.Env): The training environment.
        deployer (Deployer): The deployer agent.
        detector (Detector): The detector agent.
        memory (ReplayMemory): The replay memory.
        training (Training): The training module.
        plotter (Plotter): The plotter for visualizing results.
        episode (int): The current episode number.
        num_episodes (int): The total number of episodes.
        deployer_cumulative_reward (float): The cumulative reward of the deployer.
        detector_cumulative_reward (float): The cumulative reward of the detector.

    Returns:
        tuple: A tuple containing the updated cumulative rewards for both the deployer and the detector.
    """
    # Reset the environment and get the initial state
    state = torch.tensor(
        [env.reset()], device=DEVICE, dtype=torch.float32
    ).unsqueeze(0)

    logger.info(f"Episode {episode + 1}/{num_episodes}")

    for t in count():
        # Select actions for both deployer and detector
        deployer_action = deployer.select_action(state)
        detector_action = detector.select_action(state)

        # Execute actions in the environment and observe the next state and rewards
        next_state, (deployer_reward, detector_reward), done, _ = env.step(
            (deployer_action.item(), detector_action.item())
        )
        next_state = torch.tensor(
            [next_state], device=DEVICE, dtype=torch.float32
        )

        # Update cumulative rewards
        deployer_cumulative_reward += deployer_reward
        detector_cumulative_reward += detector_reward

        # Log information about the current step
        log_step_info(
            t,
            state,
            deployer_action,
            detector_action,
            deployer_reward,
            detector_reward,
            deployer_cumulative_reward,
            detector_cumulative_reward,
        )

        # Store the transition in replay memory
        memory.push(
            state,
            torch.tensor([[deployer_action, detector_action]], device=DEVICE),
            next_state,
            torch.tensor([deployer_reward, detector_reward], device=DEVICE),
        )

        # Update the current state
        state = next_state

        # Perform optimization step
        training.optimize_model()

        # Check if the episode is done
        if done:
            # Update the plotter with the latest rewards and plot the results
            plotter.append_rewards(
                deployer_cumulative_reward, detector_cumulative_reward
            )
            plotter.plot_rewards()
            break

    # Update the target network
    training.update_target_net()

    return deployer_cumulative_reward, detector_cumulative_reward


def log_step_info(
    t,
    state,
    deployer_action,
    detector_action,
    deployer_reward,
    detector_reward,
    deployer_cumulative_reward,
    detector_cumulative_reward,
):
    """
    Log information about the current step of the episode.

    Args:
        t (int): The current step number within the episode.
        state (torch.Tensor): The current state.
        deployer_action (int): The action taken by the deployer.
        detector_action (int): The action taken by the detector.
        deployer_reward (float): The reward received by the deployer.
        detector_reward (float): The reward received by the detector.
        deployer_cumulative_reward (float): The cumulative reward of the deployer.
        detector_cumulative_reward (float): The cumulative reward of the detector.
    """
    logger.info(f"Step {t + 1}:")
    logger.info(f"  Current State: {state.item()}")
    logger.info(
        f"  Deployer Action: {deployer_action.item()}, Reward: {deployer_reward}"
    )
    logger.info(
        f"  Detector Action: {detector_action.item()}, Reward: {detector_reward}"
    )
    logger.info(f"  Cumulative Deployer Reward: {deployer_cumulative_reward}")
    logger.info(f"  Cumulative Detector Reward: {detector_cumulative_reward}")
    logger.info("-" * 30)
