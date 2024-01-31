from marl.utils.helpers import *
from marl.utils.logger import logger


def main():
    """
    Main function to execute the training loop.
    """
    # Initialize the environment, agents, training module, memory, and plotter
    (
        env,
        attacker,
        defender,
        attacker_policy_net,
        defender_policy_net,
    ) = initialize_environment_and_agents()
    memory, training, plotter = initialize_training_memory_and_plotter(
        attacker_policy_net, defender_policy_net
    )

    num_episodes = 500
    attacker_cumulative_reward = 0
    defender_cumulative_reward = 0

    # Run training episodes
    for episode in range(num_episodes):
        attacker_cumulative_reward, defender_cumulative_reward = run_episode(
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
        )

    logger.info("Complete")
    plotter.plot_rewards(show_result=True)


if __name__ == "__main__":
    main()
