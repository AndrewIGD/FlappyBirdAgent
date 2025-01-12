
# Flappy Bird Q-Learning

This document presents a comprehensive overview of implementing and training a neural network using the Q-learning algorithm to control an agent in the game of Flappy Bird.  The implementation utilizes PyTorch for deep learning computations, OpenAI Gymnasium for managing the game environment, and the flappy_bird_gymnasium environment.

---

## **1. Architecture**
### **Q-Network**
The Q-Network is a Convolutional Neural Network (CNN) that processes stacked frames of the game environment and outputs Q-values corresponding to each possible action (`flap` or `do nothing`). This network architecture is composed of the following layers:

- **Input Layer**:
  - The input to the network is a tensor of shape 4x84x84, representing 4 consecutive grayscale frames of size 84x84 pixels.

- **Convolutional Layers**:
  1. **Conv2D Layer 1**:
     - Number of Filters: 32
     - Kernel Size: 8x8
     - Stride: 4
     - Activation Function: ReLU

  2. **Conv2D Layer 2**:
     - Number of Filters: 64
     - Kernel Size: 4x4
     - Stride: 2
     - Activation Function: ReLU

  3. **Conv2D Layer 3**:
     - Number of Filters: 64
     - Kernel Size: 3x3
     - Stride: 1
     - Activation Function: ReLU

- **Fully Connected Layers**:
  1. **Fully Connected Layer 1**:
     - Input Size: 3136 (64 x 7 x 7)
     - Output Size: 512
     - Activation Function: ReLU

  2. **Fully Connected Layer 2**:
     - Input Size: 512
     - Output Size: 2 (one for each action: `flap` or `do nothing`).

- **Output Layer**:
  - Sigmoid activation is applied to produce Q-values in the range [0, 1].

### **Replay Buffer**
The Replay Buffer is an essential component for training the agent using image frames. Instead of storing traditional state-action pairs, the Replay Buffer handles preprocessed stacked image frames to allow the agent to learn from past experiences more effectively.
 
- **Stored Data**: Each entry in the buffer consists of:
  - Stacked image frames representing the current state.
  - The action taken (`flap` or `do nothing`).
  - The reward obtained.
  - The next state (another stack of preprocessed frames).
  - A flag indicating whether the episode terminated.

- **Purpose**: By randomly sampling batches from the buffer, the agent learns from a diverse set of interactions, which:
  - Breaks correlations between consecutive frames.
  - Stabilizes the Q-value updates.

### **Epsilon-Greedy Action Selection**

The epsilon-greedy strategy is used to balance exploration (trying random actions) and exploitation (choosing the best-known actions). This mechanism ensures that the agent explores different strategies early in training and gradually focuses on optimal actions as it learns.

- **Input**: Stacked preprocessed frames representing the current state.
- **Process**:
  - With probability epsilon, the agent selects a random action (exploration).
  - Otherwise, it selects the action with the highest Q-value predicted by the Q-network (exploitation).
- **Epsilon Decay**:
  - Starts at epsilon = 0.1 (high exploration).
  - Gradually decays to epsilon = 0.0001 (minimal exploration).

This method is implemented in the `epsilon_greedy_action` function, ensuring that exploration diminishes as the agent becomes more knowledgeable.

---

## **2. Hyperparameters**

The following hyperparameters were used in the implementation, along with explanations of their roles:

1. **GAMMA**:
   - **Value**: 0.99
   - **Purpose**: Determines the importance of future rewards compared to immediate rewards. A high value (close to 1) emphasizes long-term rewards.

2. **LEARNING_RATE**:
   - **Value**: 0.0001
   - **Purpose**: Specifies the step size for updating the network weights during training. A small value ensures gradual and stable learning.

3. **BATCH_SIZE**:
   - **Value**: 32
   - **Purpose**: Number of experiences sampled from the replay buffer for each training step. Smaller sizes make updates more frequent but noisier.
4. **EPSILON_START**:
   - **Value**: 0.1
   - **Purpose**: Initial exploration probability in the epsilon-greedy policy. This controls how often the agent takes random actions.

5. **EPSILON_END**:
   - **Value**: 1e-4
   - **Purpose**: Minimum exploration probability. Ensures the agent continues to explore occasionally even in late stages of training.

6. **EPSILON_DECAY**:
   - **Value**: 0.995
   - **Purpose**: The rate at which epsilon decays after each episode, gradually reducing exploration over time.

7. **REPLAY_MEMORY_SIZE**:
   - **Value**: 50,000
   - **Purpose**: Maximum number of experiences stored in the replay buffer. This enables the agent to learn from a diverse set of past experiences.
8. **NUM_EPISODES**:
    - **Value**: 10,000
    - **Purpose**: Total number of training episodes to allow sufficient learning.

9. **FRAME_SKIP**:
   - **Value**: 4
   - **Purpose**: Number of frames skipped between actions to reduce computational load and focus on meaningful transitions.

10. **STACK_SIZE**:
    - **Value**: 4
    - **Purpose**: Number of consecutive frames stacked together as input. This provides temporal context for the agent.

