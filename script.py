import gymnasium
import flappy_bird_gymnasium
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 100
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 200
FRAME_SKIP = 4
STACK_SIZE = 4

def preprocess_frame(frame):
    frame = np.array(frame, dtype=np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame /= 255.0
    return torch.tensor(frame).float()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1920, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        if x.dim() == 4:
            for i in range(x.shape[1]):
                cv2.imwrite(f"./poze/poza{i}.png", x[0][i].detach().numpy())
        x = x.flatten(start_dim=x.dim() - 3)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


def epsilon_greedy_action(network, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = network(state_tensor)[0]
        if action > 0.5:
            action = 1
        else: action = 0

        return action

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
action_space = 1

q_network = QNetwork(STACK_SIZE, action_space).to(device)
target_network = QNetwork(STACK_SIZE, action_space).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

epsilon = EPSILON_START
total_rewards = []

def backprop():
    if len(replay_buffer) < BATCH_SIZE:
        return

    transitions = replay_buffer.sample(BATCH_SIZE)

    states = torch.stack(transitions[0]).unsqueeze(1)
    rewards = torch.tensor(transitions[2], dtype=torch.float32)

    q_values = q_network(states).flatten()

    expected_state_values = (target_network(states).max(1).values * GAMMA) + rewards

    loss = loss_fn(q_values, expected_state_values)

    loss.backward()

    torch.nn.utils.clip_grad_value_(q_network.parameters(), 100)

    optimizer.step()
    optimizer.zero_grad()

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    next_frame = pygame.surfarray.array3d(pygame.display.get_surface())
    next_frame = preprocess_frame(next_frame)

    while not done:
        actions = pygame.key.get_pressed()
        if actions[pygame.K_ESCAPE]:
            pygame.quit()
            exit(1)

        current_frame = next_frame

        action = epsilon_greedy_action(q_network, current_frame, epsilon, action_space)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_skip_reward = reward
        if action == 1:
            for _ in range(FRAME_SKIP):
                next_obs, reward, terminated, truncated, _ = env.step(0)
                total_skip_reward += reward
                if terminated or truncated:
                    break

        next_frame = pygame.surfarray.array3d(pygame.display.get_surface())
        next_frame = preprocess_frame(next_frame)

        replay_buffer.add((current_frame, action, reward, next_frame))
        total_reward += total_skip_reward
        done = terminated or truncated

        backprop()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)
    total_rewards.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

plt.plot(total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Agent Performance Over Episodes")
plt.show()
