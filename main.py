import torch
import matplotlib.pyplot as plt
from DQN_model import Qnet
from utils import ReplayBuffer, train
from env import ContinuousMazeEnv

train_dqn = False
test_dqn = True
render = True

dim_actions = 4
dim_states = 2

learning_rate = 0.0001
gamma = 0.98
buffer_limit = 50000
batch_size = 32
num_episodes = 2400

max_steps = 500

epsilon_start = 1.0     
epsilon_min = 0.05      
epsilon_decay = 0.995   
epsilon = epsilon_start

print_interval = 20

env = ContinuousMazeEnv(render_mode="human" if render else None)
q_net = Qnet(dim_actions, dim_states)
q_target = Qnet(dim_actions, dim_states)
q_target.load_state_dict(q_net.state_dict())
memory = ReplayBuffer(buffer_limit)
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)

if train_dqn:
    rewards = []
    for n_epi in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0

        for t in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = q_net.sample_action(obs_tensor, epsilon)  
            next_obs, reward, done, _, _ = env.step(action)
            memory.put((obs, action, reward, next_obs, done))

            if memory.size() > 2000:
                train(q_net, q_target, memory, optimizer, gamma, batch_size)

            obs = next_obs
            episode_reward += reward

            if render:
                env.render()
            if done:
                break

        rewards.append(episode_reward)

        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network periodically
        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"Episode: {n_epi}, Reward: {episode_reward:.2f}, Buffer Size: {memory.size()}, Epsilon: {epsilon:.3f}"
            )

        # Stop early if agent is perfect for 100 episodes
        if rewards[-100:] == [max_steps] * 100:
            print("Stopping early: agent has mastered the environment!")
            break

    torch.save(q_net.state_dict(), "qnet.pth")
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

if test_dqn:
    q_net.load_state_dict(torch.load("qnet.pth"))
    for ep in range(100):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = q_net.sample_action(obs_tensor, epsilon=0.0)  
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f"Test Episode {ep}, Total Reward: {total_reward:.2f}")

env.close()
