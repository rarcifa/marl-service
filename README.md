# Multi Agent Reinforcement Learning (MARL) Service

This repository contains a Deep Q-Network (DQN) multi agent simulation implemented in Python for reinforcement learning tasks using a custom simulated environments. The agents are designed to learn and make optimal decisions in environments with discrete action spaces.

## Prerequisites

Before running the agent, make sure you have the following dependencies installed:

- Python 3.x
- Poetry
- PyTorch
- NumPy
- Matplotlib
- OpenAI Gym

You can install Poetry and other dependencies using the following commands:

```bash
pip install poetry
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/rarcifa/marl-service
   cd marl-service
   ```

2. Initiate project:

   - Install project dependencies with poetry:

     ```bash
     poetry install
     ```

3. Run the main script:

   - Using Poetry to run the module:

     ```bash
     poetry run python -m marl
     ```

4. Using Docker (Optional):

   - Build the Docker image:

     ```bash
     docker build -t marl -f .docker/Dockerfile .
     ```

   - Run the Docker container:

     ```bash
     docker run -p 4000:80 marl
     ```

The main.py script initializes the DQN agent setup, where it trains in a simulated scenario, playing the roles of both attacker and defender. Training progress is visualized using Matplotlib.

## Agent Configuration

You can configure the agent by modifying the parameters in the `main.py` script.

Key configurations include:

- Gym environment setup for custom scenario.
- The number of episodes for training (`num_episodes`).
- Hyperparameters such as learning rate (`LR`), epsilon-greedy exploration parameters (`EPS_START`, `EPS_END`, `EPS_DECAY`), and more.
- Replay memory capacity (`ReplayMemory(10000)`).

Feel free to customize these parameters to suit your specific reinforcement learning task.

## Visualization

Training progress is visualized using Matplotlib, showing the learning curves for both the attacker and defender.

## Credits

This implementation extends the principles of Deep Q-Networks (DQN) for reinforcement learning to a cybersecurity context, focusing on smart contract vulnerabilities.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the LICENSE file for details.
