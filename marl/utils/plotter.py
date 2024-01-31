"""
Plotter Module.

This module provides the Plotter class, designed for visualizing the training progress of a reinforcement learning agent 
in scenarios involving both attackers and defenders. The class is specifically tailored to plot the cumulative rewards 
of both entities, offering insights into their performance over the course of training.

The Plotter class facilitates the visualization of the cumulative rewards for both the attacker and defender agents 
across training episodes. It enables both ongoing training visualization and final result presentation, allowing 
for an in-depth analysis of the agents' learning dynamics and strategic developments.

Classes:
    Plotter: A class for plotting and visualizing the cumulative rewards of attacker and defender agents in a 
    reinforcement learning setting.
"""

import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


class Plotter:
    """
    A class for plotting and visualizing the training progress of reinforcement learning agents.

    The Plotter class focuses on visualizing the cumulative rewards of both attacker and defender agents
    during their training process. It provides methods for updating and displaying these rewards, aiding
    in the analysis of the agents' performance across episodes.

    Attributes:
        attacker_rewards (list): A list storing the cumulative rewards of the attacker agent.
        defender_rewards (list): A list storing the cumulative rewards of the defender agent.
    """

    def __init__(self):
        """
        Initializes the Plotter with empty lists for storing attacker and defender rewards.
        """
        self.attacker_rewards = []
        self.defender_rewards = []

    def plot_rewards(
        self, show_result: bool = False, file_name: str = "training_rewards.png"
    ):
        """
        Plots the cumulative rewards of the attacker and defender.

        This method sets up a plot to visualize the cumulative rewards for both agents. It can be used to display
        the results after training or to update the plot during training.

        Args:
            show_result (bool): If True, displays the final plot. If False, updates the ongoing training plot.
            file_name (str): The filename for saving the plot.
        """
        plt.figure(2)
        plt.clf()
        plt.title("Result" if show_result else "Training...")

        # Setting labels for the axes
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")

        # Plotting the rewards for both attacker and defender
        plt.plot(self.attacker_rewards, label="Attacker")
        plt.plot(self.defender_rewards, label="Defender")

        # Add a legend to the plot
        plt.legend()

        # Pause to update the plot
        plt.pause(0.001)

        # Handling display in IPython environment
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
        # Save the plot as a PNG file
        elif show_result:
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")

    def append_rewards(self, attacker_reward, defender_reward):
        """
        Appends the latest rewards of the attacker and defender to their respective lists.

        Args:
            attacker_reward: The latest cumulative reward of the attacker.
            defender_reward: The latest cumulative reward of the defender.
        """
        self.attacker_rewards.append(attacker_reward)
        self.defender_rewards.append(defender_reward)


plt.ioff()
plt.show()
