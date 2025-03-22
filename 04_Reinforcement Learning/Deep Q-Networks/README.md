# Deep Q-Learning

## Overview & Introduction
Deep Q-Learning (DQN) is an advanced reinforcement learning algorithm that combines Q-Learning with deep neural networks to handle high-dimensional state spaces. It was introduced by DeepMind in 2013 and gained fame after mastering Atari games directly from pixel inputs.

**Role in Reinforcement Learning**:
Deep Q-Learning enables agents to learn optimal policies in complex environments where traditional tabular methods would be impractical. It bridges the gap between classical reinforcement learning and deep learning.

### Historical Context
The algorithm was first presented in the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al. in 2013. It represented a breakthrough by demonstrating that reinforcement learning could work effectively with deep neural networks, leading to human-level performance on many Atari games.

---

## Theoretical Foundations

### Conceptual Explanation
Deep Q-Learning approximates the Q-value function using a deep neural network instead of a lookup table. The Q-function maps state-action pairs to expected future rewards:

$$ Q(s, a) = \mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... | s_t = s, a_t = a] $$

where:
- $$ s $$ is the current state
- $$ a $$ is the action taken
- $$ R_t $$ is the reward at time t
- $$ \gamma $$ is the discount factor

### Mathematical Formulation

**Loss Function**: The network is trained to minimize the temporal difference error:

$$ L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2] $$

where:
- $$ \theta $$ are the parameters of the primary network
- $$ \theta^- $$ are the parameters of the target network
- $$ s' $$ is the next state
- $$ r $$ is the immediate reward

**Bellman Equation**: The optimal Q-function satisfies:

$$ Q^*(s, a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q^*(s', a') | s, a] $$

### Key Innovations

1. **Experience Replay**: Stores transitions (s, a, r, s') in a replay buffer and samples randomly to break correlations in sequential data.

2. **Target Network**: Uses a separate network with parameters $$ \theta^- $$ that are periodically updated to stabilize learning.

3. **ε-greedy Exploration**: Balances exploration and exploitation by taking random actions with probability ε.

---

## Algorithm Mechanics

### Step-by-Step Process

1. **Initialization**:
   - Initialize replay memory D with capacity N
   - Initialize primary Q-network with random weights θ
   - Initialize target Q-network with weights θ⁻ = θ

2. **Interaction Loop**:
   - For each episode:
     - Initialize state s
     - For each step:
       - Select action a using ε-greedy policy
       - Execute action a, observe reward r and next state s'
       - Store transition (s, a, r, s') in replay memory D
       - Sample random minibatch of transitions from D
       - Compute target values: y = r + γ max_a' Q(s', a'; θ⁻)
       - Update Q-network by minimizing loss: (y - Q(s, a; θ))²
       - Every C steps, update target network: θ⁻ = θ
       - s = s'

### Training & Prediction Workflow

```python
# Pseudocode for DQN training loop
def train_dqn():
    initialize_replay_memory()
    initialize_q_network()
    initialize_target_network()
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-greedy action selection
            if random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_network.predict(state)
                action = argmax(q_values)
                
            next_state, reward, done, _ = env.step(action)
            
            # Store in replay memory
            replay_memory.append((state, action, reward, next_state, done))
            
            # Sample minibatch and train
            if len(replay_memory) > batch_size:
                minibatch = random_sample(replay_memory, batch_size)
                train_on_batch(minibatch)
                
            state = next_state
            
            # Periodically update target network
            if steps % target_update_freq == 0:
                update_target_network()
```

---

## Implementation Details

### Code Structure

```python
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, batch_size=64, memory_size=10000,
                 target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        
        # Create primary Q-network
        self.q_network = self._build_model()
        
        # Create target Q-network
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(q_values)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample minibatch from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Compute Q-values for current states
        q_values = self.q_network.predict(states)
        
        # Compute Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the Q-network
        self.q_network.fit(states, q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target network periodically
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_network()
```

### Setup Instructions

```bash
pip install tensorflow numpy gym
```

---

## Hyperparameters & Optimization

Key hyperparameters that affect DQN performance:

- **Discount Factor (γ)**: Controls importance of future rewards (typical value: 0.99).
- **Exploration Rate (ε)**: Initial probability of taking random actions (typical value: 1.0).
- **Exploration Decay**: Rate at which ε decreases over time (typical value: 0.995).
- **Learning Rate**: Step size for gradient updates (typical value: 0.001).
- **Replay Memory Size**: Capacity of experience replay buffer (typical value: 10,000-1,000,000).
- **Batch Size**: Number of transitions sampled for each update (typical value: 32-128).
- **Target Network Update Frequency**: How often to update target network (typical value: 100-10,000 steps).

**Tuning Strategies**:
- Start with conservative learning rates (0.0001-0.001)
- Use larger replay buffers for complex environments
- Implement a schedule for epsilon decay

---

## Evaluation Metrics

- **Average Return**: Mean cumulative reward per episode.
- **Average Q-Value**: Mean predicted Q-value over time.
- **Learning Curve**: Plot of rewards versus training steps.
- **Success Rate**: Percentage of episodes where the agent achieves the goal.

---

## Extensions and Variants

### Double DQN
Addresses overestimation bias by decoupling action selection and evaluation:
$$ y = r + \gamma Q(s', \text{argmax}_{a'} Q(s', a'; \theta); \theta^-) $$

### Dueling DQN
Separates value and advantage functions:
$$ Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a') $$

### Prioritized Experience Replay
Samples transitions with probability proportional to TD error:
$$ P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}} $$
where $p_i$ is the priority of transition i.

---

## Practical Examples

**Environment**: Classic control problems (CartPole, MountainCar), Atari games.

**Sample Application**:
Training an agent to play Pong:
```python
import gym
from dqn import DQN
import numpy as np

# Preprocess Atari frames
def preprocess(observation):
    # Convert to grayscale, resize, normalize
    processed_obs = ...
    return processed_obs

env = gym.make("PongDeterministic-v4")
state_size = 84 * 84  # Preprocessed frame size
action_size = env.action_space.n
agent = DQN(state_size, action_size)

for episode in range(1000):
    state = preprocess(env.reset())
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess(next_state)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    print(f"Episode: {episode}, Total Reward: {total_reward}")
```

---

## Advantages & Limitations

**Advantages**:
- Can handle high-dimensional state spaces
- Learns directly from raw sensory inputs
- End-to-end learning without manual feature engineering
- Generalizes across similar states

**Limitations**:
- Sample inefficiency (requires many interactions)
- Sensitive to hyperparameter choices
- Stability issues during training
- Struggles with sparse rewards
- Overestimation bias

---

## Further Reading

1. Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." arXiv preprint arXiv:1312.5602.
2. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
3. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI Conference on Artificial Intelligence.
4. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." International Conference on Machine Learning.
5. Schaul, T., et al. (2016). "Prioritized Experience Replay." International Conference on Learning Representations.

---
