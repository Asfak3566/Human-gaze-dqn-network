# Assignment 3: DQN Agent for Continuous Maze Environment

## Description
This project implements a Deep Q-Network (DQN) agent to solve the custom continuous maze environment provided in `env.py`. The agent is trained to reach the goal while avoiding walls and danger zones, using a neural network for Q-value approximation. The environment is visualized using Pygame, and training progress is plotted with matplotlib.

## Files Included
- `main.py`           : Main script for training and testing the DQN agent
- `env.py`            : Custom continuous maze environment (do not modify wall, danger, or goal positions)
- `DQN_model.py`      : DQN neural network model
- `utils.py`          : Replay buffer and training utilities
- `qnet.pth`          : (Generated) Trained DQN model weights


## How to Run
1. **Install dependencies:**
   ```
   pip install gymnasium numpy pygame matplotlib torch
   ```
2. **Train the agent:**
   - In `main.py`, set `train_dqn = True` and `test_dqn = False`.
   - Run:
     ```
     python main.py
     ```
   - The agent will train and save the model as `qnet.pth`.
   - A training curve will be displayed and can be saved as an image.
3. **Test the agent:**
   - Set `train_dqn = False` and `test_dqn = True` in `main.py`.
   - Run:
     ```
     python main.py
     ```
 


