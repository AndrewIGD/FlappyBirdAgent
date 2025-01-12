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

print("Running on " + device.type)

testing_model = "./temp/model5400.pth"

GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 100
EPSILON_START = 0.1
EPSILON_END = 1e-4
EPSILON_DECAY = 0.995
REPLAY_MEMORY_SIZE = 50000
NUM_EPISODES = 100000
CURRENT_EPISODE = 5400
FRAME_SKIP = 4
STACK_SIZE = 4

def preprocess_frame(frame):
    frame = np.array(frame, dtype=np.float32)
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=10)
    frame = cv2.bitwise_not(frame)
    _, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return torch.tensor(frame).float().to(device)

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

class FrameStack:
    def __init__(self, stack_size):
        self.frames = deque(maxlen=stack_size)

    def add(self, frame):
        self.frames.append(frame)

    def get(self):
        return torch.stack(list(self.frames)).to(device)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=x.dim() - 3)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


def epsilon_greedy_action(network, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = network(state_tensor)

        return torch.argmax(action).item()

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
env.metadata["render_fps"] = 60
action_space = 1

q_network = QNetwork().to(device)

test_mode = False
if testing_model != "":
    q_network.load_state_dict(torch.load(testing_model))
    #q_network.eval()
    #test_mode = True
    print("Running model from path: " + testing_model)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
frame_stack = FrameStack(STACK_SIZE)

epsilon = EPSILON_START
total_rewards = []

obs, _ = env.reset()

next_frame = pygame.surfarray.array3d(pygame.display.get_surface())
next_frame = preprocess_frame(next_frame)

for i in range(STACK_SIZE):
    frame_stack.add(next_frame)

def backprop():
    if len(replay_buffer) < BATCH_SIZE:
        return

    transitions = replay_buffer.sample(BATCH_SIZE)

    states = torch.stack(transitions[0]).to(device)
    actions = torch.from_numpy(np.array(transitions[1])).to(device)
    rewards = torch.tensor(transitions[2], dtype=torch.float32).to(device)
    next_states = torch.stack(transitions[3]).to(device)
    dones = torch.tensor(transitions[4], dtype=torch.bool).to(device)

    q_values = torch.sum(q_network(states) * actions, dim=1)

    next_state_values = q_network(next_states)

    expected_state_values = torch.stack(tuple(
        next_state_reward if next_state_done else next_state_reward + GAMMA * torch.max(next_state_value) for next_state_value, next_state_reward, next_state_done in zip(next_state_values, rewards, dones)
    ))

    loss = loss_fn(q_values, expected_state_values)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

for episode in range(CURRENT_EPISODE, NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    if episode % 200 == 0:
        torch.save(q_network.state_dict(), f"model{episode}.pth")

    while not done:
        actions = pygame.key.get_pressed()
        if actions[pygame.K_ESCAPE]:
            pygame.quit()
            exit(1)

        current_state = frame_stack.get()

        action = epsilon_greedy_action(q_network, current_state, epsilon, action_space)

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

        frame_stack.add(next_frame)

        next_state = frame_stack.get()

        done = terminated or truncated
        replay_buffer.add((current_state, [1, 0] if action == 0 else [0, 1], reward, next_state, done))
        total_reward += total_skip_reward

        if not test_mode:
            backprop()

    epsilon = max(EPSILON_END, EPSILON_END + (EPSILON_START - EPSILON_END) / NUM_EPISODES * (NUM_EPISODES - episode))
    print("Epsion " + str(epsilon))
    total_rewards.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

plt.plot(total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Agent Performance Over Episodes")
plt.show()
