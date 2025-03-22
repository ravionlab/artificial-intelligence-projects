# Policy Gradient Methods

## Overview & Introduction
Policy Gradient methods are a family of reinforcement learning algorithms that directly optimize the policy by computing gradients of expected return with respect to policy parameters. Unlike value-based methods, they learn a parameterized policy that directly maps states to actions.

**Role in Reinforcement Learning**:
Policy Gradient methods excel in continuous action spaces and problems where the optimal policy is stochastic. They form the foundation for many state-of-the-art reinforcement learning algorithms and have been successfully applied to robotics, game playing, and complex control tasks.

### Historical Context
Policy Gradient methods have roots dating back to the 1990s, with key developments by Williams (REINFORCE, 1992) and Sutton et al. (Policy Gradient Theorem, 1999). They have evolved into more sophisticated algorithms like A2C, TRPO, and PPO, which address the high variance and sample inefficiency of early approaches.

---

## Theoretical Foundations

### Conceptual Explanation
The core idea is to parameterize the policy $\pi_\theta(a|s)$ with parameters $\theta$ and update these parameters in the direction that maximizes expected returns. The policy directly outputs action probabilities (for discrete actions) or probability distributions (for continuous actions).

### Mathematical Formulation

**Policy Gradient Theorem**:
The gradient of the expected return with respect to policy parameters is:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)] $$

where:
- $J(\theta)$ is the expected return
- $\pi_\theta(a|s)$ is the probability of taking action $a$ in state $s$ under policy $\pi_\theta$
- $Q^{\pi_\theta}(s,a)$ is the action-value function under policy $\pi_\theta$

**REINFORCE Algorithm**:
The basic policy gradient algorithm uses Monte Carlo sampling to estimate the gradient:

$$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \cdot G_t^i $$

where:
- $G_t^i$ is the return from timestep $t$ in episode $i$
- $N$ is the number of episodes
- $T$ is the episode length

### Variance Reduction Techniques

**Baseline Subtraction**:
To reduce variance, a baseline function $b(s)$ is subtracted:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot (Q^{\pi_\theta}(s,a) - b(s))] $$

The baseline does not introduce bias if it is independent of actions.

**Advantage Function**:
Using the advantage function $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$ further reduces variance:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)] $$

---

## Algorithm Mechanics

### Step-by-Step Process

1. **Policy Parameterization**:
   - Define a policy function $\pi_\theta(a|s)$ (e.g., neural network)
   - Initialize parameters $\theta$

2. **Trajectory Collection**:
   - Sample trajectories $\tau = (s_0, a_0, r_0, s_1, ..., s_T)$ by executing policy $\pi_\theta$
   - Compute returns $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$

3. **Gradient Computation**:
   - Calculate policy gradient using sampled trajectories
   - For REINFORCE: $\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$

4. **Parameter Update**:
   - Update policy parameters using gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

5. **Repeat**:
   - Return to step 2 until convergence

### Training & Prediction Workflow

```python
# Pseudocode for basic REINFORCE algorithm
def train_policy_gradient():
    initialize_policy_parameters()
    
    for episode in range(num_episodes):
        # Collect trajectory
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            # Sample action from policy
            action_probs = policy_network(state)
            action = sample_action(action_probs)
            
            # Execute action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # Compute returns
        returns = compute_returns(rewards)
        
        # Update policy
        update_policy(states, actions, returns)
```

---

## Implementation Details

### Code Structure

```python
import numpy as np
import tensorflow as tf
import gym

class PolicyGradient:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Build policy network
        self.model = self._build_policy_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def _build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model
    
    def select_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state)[0]
        return np.random.choice(self.action_dim, p=action_probs)
    
    def compute_returns(self, rewards):
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        # Normalize returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def update_policy(self, states, actions, returns):
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(returns)
        
        with tf.GradientTape() as tape:
            # Forward pass through the network
            action_probs = self.model(states, training=True)
            
            # Create one-hot encoding of actions
            action_masks = tf.one_hot(actions, self.action_dim)
            
            # Select the probabilities of the taken actions
            selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
            
            # Compute log probabilities
            log_probs = tf.math.log(selected_action_probs)
            
            # Weight log probabilities by returns
            loss = -tf.reduce_sum(log_probs * returns)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()
    
    def train(self, env, num_episodes=1000):
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset variables
            states, actions, rewards = [], [], []
            state = env.reset()
            done = False
            total_reward = 0
            
            # Collect trajectory
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            # Compute returns
            returns = self.compute_returns(rewards)
            
            # Update policy
            loss = self.update_policy(states, actions, returns)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {total_reward}, Loss: {loss}")
        
        return episode_rewards
```

### Setup Instructions

```bash
pip install tensorflow numpy gym
```

---

## Hyperparameters & Optimization

Key hyperparameters that affect Policy Gradient performance:

- **Learning Rate**: Step size for gradient updates (typical value: 0.01-0.001).
- **Discount Factor (γ)**: Controls importance of future rewards (typical value: 0.99).
- **Network Architecture**: Number of layers and neurons in policy network.
- **Batch Size**: Number of episodes per update (larger batches reduce variance).
- **Entropy Regularization**: Encourages exploration by penalizing deterministic policies.

**Tuning Strategies**:
- Start with conservative learning rates (0.001)
- Use larger batch sizes for high-variance environments
- Add entropy regularization to prevent premature convergence
- Normalize observations and rewards

---

## Evaluation Metrics

- **Average Episode Return**: Mean cumulative reward per episode.
- **Learning Curve**: Plot of episode returns over training time.
- **Policy Entropy**: Measures exploration (higher values indicate more exploration).
- **Variance of Gradients**: Indicates stability of training process.

---

## Variants and Extensions

### REINFORCE with Baseline
Adds a state-value function to reduce variance:

```python
def update_policy(self, states, actions, returns):
    # Baseline value predictions
    baseline_values = self.value_network(states)
    
    # Advantage estimates
    advantages = returns - baseline_values
    
    # Rest of update is the same as basic REINFORCE but using advantages
    # instead of returns
```

### Actor-Critic Methods
Combines policy gradient with value function approximation:

- **Actor**: Policy network that selects actions
- **Critic**: Value network that evaluates states or state-action pairs

### Natural Policy Gradient
Uses the Fisher information matrix to ensure more stable updates:

$$ \theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta) $$

where $F$ is the Fisher information matrix.

---

## Practical Examples

**Environments**: CartPole, MountainCar, Atari games, robotics tasks.

**Sample Application**:
Training an agent to solve the CartPole problem:

```python
import gym
from policy_gradient import PolicyGradient

# Create environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize agent
agent = PolicyGradient(state_dim, action_dim)

# Train agent
rewards = agent.train(env, num_episodes=500)

# Test the trained policy
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Test episode reward: {total_reward}")
```

---

## Advantages & Limitations

**Advantages**:
- Directly optimizes the policy
- Naturally handles continuous action spaces
- Can learn stochastic policies
- Converges to local optima (at least)
- Well-suited for problems with simple optimal policies

**Limitations**:
- High variance in gradient estimates
- Sample inefficiency (requires many environment interactions)
- Sensitive to hyperparameter choices
- Can converge to suboptimal local optima
- Struggles with long-horizon tasks and sparse rewards

---

## Further Reading

1. Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning, 8(3-4), 229-256.
2. Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). "Policy gradient methods for reinforcement learning with function approximation." NIPS.
3. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). "Trust region policy optimization." ICML.
4. Kakade, S. M. (2002). "A natural policy gradient." NIPS.
5. Thomas, P. (2014). "Bias in natural actor-critic algorithms." ICML.

---
