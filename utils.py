import torch
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = zip(*mini_batch)

        return (torch.tensor(s_lst, dtype=torch.float32),
                torch.tensor(a_lst, dtype=torch.int64),
                torch.tensor(r_lst, dtype=torch.float32),
                torch.tensor(s_prime_lst, dtype=torch.float32),
                torch.tensor(done_lst, dtype=torch.float32))

    def size(self):
        return len(self.buffer)

def train(q, q_target, memory, optimizer, gamma, batch_size):
    s, a, r, s_prime, done = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1, a.unsqueeze(1)).squeeze(1)

    max_q_prime = q_target(s_prime).max(1)[0]
    target = r + gamma * max_q_prime * (1 - done)

    loss = torch.nn.functional.smooth_l1_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
