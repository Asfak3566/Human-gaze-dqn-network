# Human-gaze-dqn-network
Deep Q-Network for human gaze prediction and simulation.
=====================================================
 Human Gaze DQN Network
=====================================================

This project implements a Deep Q-Network (DQN) for human gaze
prediction and simulation. The model uses reinforcement learning
techniques to predict human gaze behaviors in dynamic environments.

-----------------------------------------------------
 Overview
-----------------------------------------------------
The Human Gaze DQN Network trains an agent to understand and
anticipate human gaze directions. This contributes to research
in human-computer interaction, attention modeling, and adaptive
user interfaces.

-----------------------------------------------------
 Project Structure
-----------------------------------------------------
human-gaze-dqn-network/
├── DQN_model.py        # Defines the Deep Q-Network architecture
├── env.py              # Custom environment for gaze prediction
├── main.py             # Main script to train and evaluate the model
├── utils.py            # Utility functions for data and model handling
├── Figure_1.png        # Visual representation of the model
├── dqn_maze.pth        # Pre-trained model weights
├── qnet.pth            # Additional model weights
├── README.md           # Markdown documentation
└── README.txt          # This file

-----------------------------------------------------
 Installation
-----------------------------------------------------
1. Clone the repository:
   git clone https://github.com/Asfak3566/Human-gaze-dqn-network.git
   cd Human-gaze-dqn-network

2. Install dependencies:
   pip install -r requirements.txt

*Note: Python 3.10+ recommended.*

-----------------------------------------------------
 Usage
-----------------------------------------------------
1. Train the model:
   python main.py

2. Evaluate / test the model:
   python evaluate.py

*Ensure that the trained model weights are correctly loaded.*

-----------------------------------------------------
 Model Architecture
-----------------------------------------------------
- Convolutional layers to process visual input
- Fully connected layers to output Q-values for gaze actions
- Designed to predict human gaze based on environmental stimuli
  
 Acknowledgments
-----------------------------------------------------
- Inspired by research in human gaze prediction and reinforcement learning
- Utilizes PyTorch and OpenAI Gym for model development and simulation
