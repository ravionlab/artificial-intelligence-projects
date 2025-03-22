# Q-Learning

## Overview & Introduction
Q-Learning is a foundational off-policy reinforcement learning algorithm that learns to make optimal decisions by estimating the value of taking actions in given states. It is a model-free approach that can learn directly from interactions with the environment without requiring a model of the environment's dynamics.

**Role in Reinforcement Learning**:
Q-Learning serves as the cornerstone for value-based reinforcement learning methods. It directly approximates the optimal action-value function, which tells the agent the expected return of taking a specific action in a specific state and following the optimal policy thereafter.

### Historical Context
Developed by Chris Watkins in 1989 during his PhD thesis, Q-Learning was one of the first algorithms to provide guarantees of convergence to optimal policies in tabular settings. Its simplicity and effectiveness have made it a fundamental algorithm in reinforcement learning, leading to numerous extensions and improvements including Deep Q-Networks (DQN).

---

## Theoretical Foundations

### Conceptual Explanation
Q-Learning maintains a table (or function approximator) of Q-values for each state-action pair, representing the expected cumulative future reward when taking action a in state s and following the optimal policy thereafter. The agent updates these Q-values through interactions with the environment, gradually improving its estimate of the optimal policy.

### Mathematical Formulation

**Q-Value Definition**:
The Q-value is defined as the expected return starting from state s, taking action a, and thereafter following policy π:

$$ Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right] $$

where:
- γ is the discount factor
- R is the reward
- π is the policy

**Bellman Optimality Equation**:
The optimal Q-function satisfies:

$$ Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a \right] $$

**Q-Learning Update Rule**:
The core of Q-Learning is the update equation:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$

where:
- α is the learning rate
- $r_{t+1}$ is the immediate reward
- $\gamma \max_{a} Q(s_{t+1}, a)$ is the bootstrapped estimate of future rewards
- The term in brackets is known as the temporal-difference (TD) error

### Exploration vs. Exploitation

Q-Learning typically uses an ε-greedy policy for action selection:
- With probability ε: Select a random action (exploration)
- With probability 1-ε: Select action with highest Q-value (exploitation)

The value of ε is often annealed over time to transition from exploration to exploitation.

---

## Algorithm Mechanics

### Step-by-Step Process

1. **Initialization**:
   - Initialize Q(s,a) arbitrarily for all state-action pairs
   - Set learning rate α, discount factor γ, and exploration rate ε

2. **For each episode**:
   - Initialize state s
   - For each step of episode:
     - Choose action a from s using policy derived from Q (e.g., ε-greedy)
     - Take action a, observe reward r and next state s'
     - Update Q(s,a) using the Q-learning update rule
     - s ← s'
     - If s is terminal, end episode

### Training & Prediction Workflow

```python
# Pseudocode for Q-Learning algorithm
def q_learning():
    initialize_q_table()  # Initialize Q(s,a) for all s∈S, a∈A(s)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Select action using ε-greedy policy
            if random() < epsilon:
                action = random_action()
            else:
                action = argmax(Q[state,:])
                
            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value
            td_target = reward + gamma * max(Q[next_state,:]) * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
```

---

## Implementation Details

### Code Structure

```python
import numpy as np
import gym

class QLearning:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))
    
    def select_action(self, state):
        # ε-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])
    
    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        
    def decay_epsilon(self):
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def train(self, env, num_episodes):
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
            
            self.decay_epsilon()
            rewards.append(total_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        return rewards
```

### Setup Instructions

```bash
pip install numpy gym
```

---

## Hyperparameters & Optimization

Key hyperparameters that affect Q-Learning performance:

- **Learning Rate (α)**: Controls how much new information overrides old information (typical value: 0.01-0.1).
- **Discount Factor (γ)**: Determines importance of future rewards (typical value: 0.9-0.99).
- **Initial Exploration Rate (ε)**: Starting probability of taking random actions (typical value: 1.0).
- **Exploration Decay Rate**: Rate at which ε decreases (typical value: 0.995-0.999).
- **Minimum Exploration Rate**: Lower bound for ε (typical value: 0.01-0.1).

**Tuning Strategies**:
- Use higher learning rates for simple environments, lower for complex ones
- Adjust exploration decay based on number of episodes
- For deterministic environments, γ can be closer to 1.0
- For stochastic environments, consider lower α to average over randomness

---

## Evaluation Metrics

- **Average Return**: Mean episodic reward over multiple episodes.
- **Learning Curve**: Plot of episodic rewards over training time.
- **State Visitation Frequency**: Distribution of states visited during training.
- **Policy Quality**: Optimality of the learned policy compared to known optimal solutions.
- **Convergence Speed**: Number of episodes until stable performance.

---

## Extensions and Variants

### Double Q-Learning
Addresses overestimation bias by using two Q-functions:

$$ Q_1(s_t, a_t) \leftarrow Q_1(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q_2(s_{t+1}, \arg\max_a Q_1(s_{t+1}, a)) - Q_1(s_t, a_t) \right] $$

### Expected SARSA
Uses expected value over all possible next actions:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \sum_a \pi(a|s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$

### Q-Learning with Function Approximation
Uses function approximators (e.g., linear models, neural networks) to generalize across states:

$$ Q(s, a; \theta) \approx Q^*(s, a) $$

Where θ are the parameters of the function approximator.

---

## Practical Examples

**Environments**: Grid World, FrozenLake, Taxi, MountainCar, Cliff Walking.

**Sample Application**:
Training an agent to navigate the FrozenLake environment:

```python
import gym
import numpy as np
from q_learning import QLearning

# Create environment
env = gym.make('FrozenLake-v1')
state_size = env.observation_space.n
action_size = env.action_space.n

# Create agent
agent = QLearning(state_size, action_size, alpha=0.1, gamma=0.99, 
                  epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

# Train agent
rewards = agent.train(env, num_episodes=5000)

# Test learned policy
def test_policy(env, agent, num_episodes=100):
    successes = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.q_table[state, :])  # Greedy action
            state, reward, done, _ = env.step(action)
            if reward > 0:
                successes += 1
    
    success_rate = successes / num_episodes
    print(f"Success rate: {success_rate:.2f}")
    return success_rate

test_policy(env, agent)
```

---

## Advantages & Limitations

**Advantages**:
- Simple to understand and implement
- Works well in discrete, fully observable environments
- Guaranteed convergence to optimal policy in tabular settings
- Off-policy learning enables learning from any experience
- Forms the foundation for more advanced algorithms

**Limitations**:
- Curse of dimensionality (table size grows exponentially with state/action spaces)
- Struggles with continuous state/action spaces without function approximation
- Sample inefficiency (requires many environment interactions)
- Slow convergence on large problems
- Bootstrapping can lead to instability with function approximation
- Overestimation bias due to max operator

---

## Further Reading

1. Watkins, C. J. C. H. (1989). "Learning from Delayed Rewards." PhD Thesis, Cambridge University.
2. Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction." MIT Press.
3. van Hasselt, H. (2010). "Double Q-learning." NIPS.
4. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
5. Hasselt, H. V., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

---
