import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ContinuousMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.width = 600
        self.height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.step_size = 0.05
        self.agent_pos = None

        self.goal_pos = np.array([0.9, 0.5], dtype=np.float32)
        self.goal_radius = 0.05

        self.danger_zones = [(0.4, 0.85, 0.6, 0.9),
                             (0.4, 0.1, 0.6, 0.15),
                             (0.45, 0.48, 0.55, 0.52)]

        self.walls = [(0.3, 0.9, 0.7, 1.0),
                      (0.3, 0.0, 0.7, 0.1)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0.1, 0.5], dtype=np.float32)
        return self.agent_pos.copy(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        delta = np.zeros_like(self.agent_pos)
        if action == 0: delta[1] = self.step_size
        elif action == 1: delta[1] = -self.step_size
        elif action == 2: delta[0] = -self.step_size
        elif action == 3: delta[0] = self.step_size

        new_pos = np.clip(self.agent_pos + delta, 0.0, 1.0)
        collided = any(xmin <= new_pos[0] <= xmax and ymin <= new_pos[1] <= ymax for (xmin, ymin, xmax, ymax) in self.walls)

        reward = -0.01
        done = False
        if collided:
            reward = -0.1
            new_pos = self.agent_pos.copy()
        else:
            self.agent_pos = new_pos

        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            if xmin <= self.agent_pos[0] <= xmax and ymin <= self.agent_pos[1] <= ymax:
                reward -= 10.0
                done = True
                break

        if np.linalg.norm(self.agent_pos - self.goal_pos) <= self.goal_radius:
            reward += 10.0
            done = True

        return self.agent_pos.copy(), reward, done, False, {}

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Continuous Maze")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        for (xmin, ymin, xmax, ymax) in self.walls:
            rect = pygame.Rect(xmin * self.width, self.height - ymax * self.height,
                               (xmax - xmin) * self.width, (ymax - ymin) * self.height)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            rect = pygame.Rect(xmin * self.width, self.height - ymax * self.height,
                               (xmax - xmin) * self.width, (ymax - ymin) * self.height)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        pygame.draw.circle(self.screen, (0, 255, 0),
                           (int(self.goal_pos[0] * self.width), int(self.height - self.goal_pos[1] * self.height)),
                           int(self.goal_radius * self.width))

        pygame.draw.circle(self.screen, (0, 0, 255),
                           (int(self.agent_pos[0] * self.width), int(self.height - self.agent_pos[1] * self.height)), 10)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
