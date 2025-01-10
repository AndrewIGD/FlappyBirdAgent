import gymnasium
import flappy_bird_gymnasium
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import matplotlib.pyplot as plt

# Setare dispozitiv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parametrii globali
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 200
FRAME_SKIP = 4
STACK_SIZE = 4

# Preprocesare imagine
def preprocess_frame(frame):
    # Verifică dacă frame-ul are mai mult de 1 canal (RGB sau RGBA)
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # Imagine RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:  # Imagine RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    # Redimensionare la dimensiunea dorită (84x84, de exemplu)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    # Normalizare la intervalul [0, 1]
    frame = frame / 255.0

    return frame
# Stivuire cadre (4 cadre consecutive)
def stack_frames(frames, new_frame, stack_size=4):
    # Dacă frames este gol, inițializează cu cadre goale
    if frames is None:
        frames = np.zeros((stack_size, new_frame.shape[0], new_frame.shape[1]))

    # Asigură-te că new_frame are dimensiunea corectă
    if len(new_frame.shape) == 2:  # Dacă e 2D, adaugă o dimensiune suplimentară
        new_frame = np.expand_dims(new_frame, axis=0)

    # Actualizează stiva de cadre
    frames = np.append(frames[1:], new_frame, axis=0)

    return frames


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# Rețea neuronală (CNN)
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Funcție pentru alegerea acțiunii (epsilon-greedy)
def epsilon_greedy_action(network, state, epsilon, action_space):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = network(state_tensor)
        return q_values.argmax().item()

# Funcție pentru antrenarea Q-learning
def train_dqn(q_network, target_network, optimizer, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    next_q_values = target_network(next_states_tensor).max(1)[0]
    target_q_values = rewards_tensor + GAMMA * next_q_values * (1 - dones_tensor)

    loss = nn.MSELoss()(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main loop
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
action_space = env.action_space.n

q_network = QNetwork(STACK_SIZE, action_space).to(device)
target_network = QNetwork(STACK_SIZE, action_space).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

epsilon = EPSILON_START
total_rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    frame = preprocess_frame(obs)
    frames = stack_frames(None, frame)
    done = False
    total_reward = 0

    while not done:
        actions = pygame.key.get_pressed()
        if actions[pygame.K_ESCAPE]:
            pygame.quit()
            exit(1)

        action = epsilon_greedy_action(q_network, frames, epsilon, action_space)
        #next_obs, reward, terminated, truncated, _ = env.step(action)
        #next_frame = preprocess_frame(next_obs)
        #next_frames = stack_frames(frames, next_frame)
        total_reward_for_skip = 0
        for _ in range(FRAME_SKIP):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward_for_skip += reward
            if terminated or truncated:
                break
        next_frame = preprocess_frame(next_obs)
        next_frames = stack_frames(frames, next_frame)

        replay_buffer.add((frames, action, reward, next_frames, terminated))
        frames = next_frames
        total_reward += total_reward_for_skip
        done = terminated or truncated

        train_dqn(q_network, target_network, optimizer, replay_buffer)

    if episode % TARGET_UPDATE_FREQ == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)
    total_rewards.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

# Grafic performanță
plt.plot(total_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Agent Performance Over Episodes")
plt.show()
