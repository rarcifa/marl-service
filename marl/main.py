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


def process_state(state_dict):
    # Flatten the state dictionary into a list or array of numeric values
    # Example implementation could involve concatenating values or more complex processing
    processed_state = [
        state_dict["externalCalls"],
        state_dict["stateUpdates"],
        *state_dict["functionTypes"],
        state_dict["pattern_presence"]["call_value"],
        state_dict["pattern_presence"]["delegatecall"],
        state_dict["pattern_presence"]["selfdestruct"],
    ]
    return processed_state


def main():
    """
    Main function to execute the training loop, with all logic included directly.
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
    n_observations = 9

    # Initialize DQN models for both deployer and detector
    deployer_policy_net = DQN(n_observations, n_actions).to(DEVICE)
    detector_policy_net = DQN(n_observations, n_actions).to(DEVICE)

    # Create deployer and detector agents
    deployer = Deployer(deployer_policy_net, EPS_START, EPS_END, EPS_DECAY)
    detector = Detector(detector_policy_net, EPS_START, EPS_END, EPS_DECAY)

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

    num_episodes = 6000
    deployer_cumulative_reward = 0
    detector_cumulative_reward = 0

    # Run training episodes
    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        raw_initial_state = env.reset()

        # Process the raw state to get it into the correct numeric format
        processed_initial_state = process_state(
            raw_initial_state
        )  # This should be defined based on your state structure

        # Convert the processed state into a tensor
        state = torch.tensor(
            [processed_initial_state], device=DEVICE, dtype=torch.float32
        )

        logger.info(f"Episode {episode + 1}/{num_episodes}")

        for t in count():
            # Select actions for both deployer and detector
            deployer_action = deployer.select_action(state)
            detector_action = detector.select_action(state)

            # Execute actions in the environment and observe the next state and rewards
            raw_next_state, (deployer_reward, detector_reward), done, _ = (
                env.step((deployer_action.item(), detector_action.item()))
            )

            # Process the raw next state
            processed_next_state = process_state(raw_next_state)

            # Convert the processed next state into a tensor
            next_state = torch.tensor(
                [processed_next_state], device=DEVICE, dtype=torch.float32
            )

            # Update cumulative rewards
            deployer_cumulative_reward += deployer_reward
            detector_cumulative_reward += detector_reward

            # Log information about the current step
            logger.info(f"Step {t + 1}:")
            logger.info(f"  Current State: {state}")
            logger.info(
                f"  Deployer Action: {deployer_action.item()}, Reward: {deployer_reward}"
            )
            logger.info(
                f"  Detector Action: {detector_action.item()}, Reward: {detector_reward}"
            )
            logger.info(
                f"  Cumulative Deployer Reward: {deployer_cumulative_reward}"
            )
            logger.info(
                f"  Cumulative Detector Reward: {detector_cumulative_reward}"
            )
            logger.info("-" * 30)

            # Store the transition in replay memory
            memory.push(
                state,
                torch.tensor(
                    [[deployer_action, detector_action]], device=DEVICE
                ),
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

    logger.info("Complete")
    plotter.plot_rewards(show_result=True)


if __name__ == "__main__":
    main()
