#!/usr/bin/env python3
"""
Advanced AI training example using Deep Q-Learning (DQN).
This demonstrates GPU-accelerated training on Jetson Nano.
"""

import random
from collections import deque

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("PyTorch not found. Install it for deep learning training.")
    print("On Jetson Nano, follow: https://forums.developer.nvidia.com/t/pytorch-for-jetson")

# Import CartPole
try:
    import cartpole

    HAS_CARTPOLE = True
except ImportError:
    HAS_CARTPOLE = False
    print("CartPole module not found. Build with: cargo build --release --features python")


class DQN(nn.Module):
    """Deep Q-Network for CartPole"""

    def __init__(self, state_size=4, action_size=2, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for CartPole"""

    def __init__(self, device='cpu'):
        self.device = device
        self.state_size = 4
        self.action_size = 2

        # Hyperparameters (tuned for better learning)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Keep exploring more
        self.epsilon_decay = 0.998  # Decay slower
        self.learning_rate = 0.0005  # Reduced for stability
        self.batch_size = 64

        # Networks
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_dqn(episodes=500, train_freq=4):
    """Train DQN agent on CartPole

    Args:
        episodes: Number of episodes to train
        train_freq: Train every N steps (higher = faster but less stable)
    """

    if not HAS_PYTORCH or not HAS_CARTPOLE:
        print("\nMissing dependencies. Cannot run training.")
        return None

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"Training DQN Agent on CartPole")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Episodes: {episodes}")
    print(f"{'=' * 60}\n")

    # Create environment and agent
    env = cartpole.PyCartPole()
    agent = DQNAgent(device=device)

    # Training metrics
    episode_rewards = []
    losses = []

    # Timing
    import time
    start_time = time.time()

    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        done = False
        steps = 0

        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # Reward shaping: encourage staying upright and centered
            # state = [x, x_dot, theta, theta_dot]
            x, x_dot, theta, theta_dot = next_state

            # Penalty for being far from center
            distance_penalty = abs(x) * 0.1

            # Penalty for large angle
            angle_penalty = abs(theta) * 2.0

            # Shaped reward
            shaped_reward = reward - distance_penalty - angle_penalty

            # Big penalty for failure
            if done:
                shaped_reward = -10.0

            # Store transition with shaped reward
            agent.memory.push(state, action, shaped_reward, next_state, float(done))

            # Train every train_freq steps (faster training)
            if steps % train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state
            total_reward += reward  # Track original reward
            steps += 1

        # Update target network
        if episode % 10 == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(total_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Print progress
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta_seconds = (episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0

            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Avg (10): {avg_reward:6.1f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Speed: {eps_per_sec:.1f} eps/s | "
                  f"ETA: {int(eta_seconds)}s")

        # Check if solved
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= 195:
                print(f"\n{'=' * 60}")
                print(f"Solved in {episode + 1} episodes!")
                print(f"Average reward over last 100 episodes: {avg_100:.1f}")
                print(f"{'=' * 60}\n")
                break

    # Final statistics
    total_time = time.time() - start_time
    avg_eps_per_sec = len(episode_rewards) / total_time

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"{'=' * 60}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Training time: {int(total_time)}s ({int(total_time / 60)}m {int(total_time % 60)}s)")
    print(f"Speed: {avg_eps_per_sec:.1f} episodes/second")
    print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"Best episode: {max(episode_rewards):.1f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"{'=' * 60}\n")

    # Save model
    print("Saving model...")
    torch.save(agent.policy_net.state_dict(), "cartpole_dqn.pth")
    print("Model saved to: cartpole_dqn.pth")

    return agent, episode_rewards


def test_trained_agent(model_path="cartpole_dqn.pth", episodes=10):
    """Test a trained agent"""

    if not HAS_PYTORCH or not HAS_CARTPOLE:
        print("\nMissing dependencies. Cannot run test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    agent = DQNAgent(device=device)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0  # No exploration

    print(f"\n{'=' * 60}")
    print(f"Testing Trained Agent")
    print(f"{'=' * 60}\n")

    env = cartpole.PyCartPole()
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.1f} reward, {steps} steps")

    print(f"\n{'=' * 60}")
    print(f"Average reward: {np.mean(rewards):.1f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    if not HAS_PYTORCH:
        print("\nPlease install PyTorch to run this example.")
        sys.exit(1)

    if not HAS_CARTPOLE:
        print("\nPlease build the CartPole module first:")
        print("  cargo build --release --features python")
        print("  cp target/release/libcartpole.so cartpole.so  # Linux")
        print("  copy target\\release\\cartpole.pyd cartpole.pyd  # Windows")
        sys.exit(1)

    # Train the agent
    train_dqn(episodes=500)

    # Test the trained agent
    print("\nTesting trained agent...")
    test_trained_agent(episodes=5)
