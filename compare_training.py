#!/usr/bin/env python3
"""
Compare Simple AI vs DQN training progress
"""

import os
import sys

# Check dependencies
try:
    import torch
    import numpy as np

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("PyTorch not found.")
    sys.exit(1)

try:
    import cartpole

    HAS_CARTPOLE = True
except ImportError:
    HAS_CARTPOLE = False
    print("CartPole module not found.")
    sys.exit(1)

from train_dqn import DQNAgent


def test_agent(agent_type="simple", model_path=None, episodes=10):
    """Test an agent and return average reward"""
    env = cartpole.PyCartPole()
    rewards = []

    if agent_type == "simple":
        # Simple random baseline
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = 1 if state[2] > 0 else 0  # Simple policy: lean opposite to angle
                state, reward, done = env.step(action)
                total_reward += reward
            rewards.append(total_reward)

    elif agent_type == "dqn" and model_path:
        # DQN agent
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(device=device)

        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
            agent.policy_net.eval()
            agent.epsilon = 0  # No exploration

            for _ in range(episodes):
                state = env.reset()
                total_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state)
                    state, reward, done = env.step(action)
                    total_reward += reward
                rewards.append(total_reward)
        else:
            print(f"Model not found: {model_path}")
            return None

    return np.mean(rewards) if rewards else None


def main():
    print("\n" + "=" * 60)
    print("Comparing AI Performance")
    print("=" * 60)

    # Test simple baseline
    print("\nTesting simple baseline agent...")
    simple_avg = test_agent("simple", episodes=20)
    print(f"Simple Agent: {simple_avg:.1f} average reward")

    # Test DQN if model exists
    if os.path.exists("cartpole_dqn.pth"):
        print("\nTesting trained DQN agent...")
        dqn_avg = test_agent("dqn", "cartpole_dqn.pth", episodes=20)
        if dqn_avg:
            print(f"DQN Agent:    {dqn_avg:.1f} average reward")

            print("\n" + "=" * 60)
            if dqn_avg > simple_avg:
                improvement = ((dqn_avg - simple_avg) / simple_avg) * 100
                print(f"✓ DQN is {improvement:.1f}% better than baseline!")
            else:
                decline = ((simple_avg - dqn_avg) / simple_avg) * 100
                print(f"✗ DQN is {decline:.1f}% worse than baseline")
                print("  Try training longer or adjusting hyperparameters")
            print("=" * 60)
    else:
        print("\nNo trained model found. Train with: python3.6 train_dqn.py")

    print()


if __name__ == "__main__":
    main()
