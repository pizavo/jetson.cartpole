#!/usr/bin/env python3.6
"""
Plot training progress from saved models
Shows how the DQN improves over time
"""

import os
import sys

try:
    import torch
    import numpy as np
except ImportError:
    print("NumPy/PyTorch not available")
    sys.exit(1)

try:
    import cartpole
except ImportError:
    print("CartPole module not found")
    sys.exit(1)


def test_model_performance(model_path, episodes=20):
    """Test a saved model and return average performance"""
    from train_dqn import DQNAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)

    if not os.path.exists(model_path):
        return None

    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()
    agent.epsilon = 0  # No exploration

    env = cartpole.PyCartPole()
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards), min(rewards), max(rewards)


def simple_bar_chart(values, labels, width=50):
    """Create a simple ASCII bar chart"""
    max_val = max(values)
    print("\n" + "="*70)
    print("Performance Comparison")
    print("="*70)

    for val, label in zip(values, labels):
        bar_len = int((val / max_val) * width)
        bar = "█" * bar_len
        print(f"{label:20s} | {bar} {val:.1f}")

    print("="*70)
    print(f"Goal: 195.0 reward (considered 'solved')")
    print("="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("DQN Training Progress Analysis")
    print("="*70 + "\n")

    # Check if model exists
    if not os.path.exists("cartpole_dqn.pth"):
        print("No trained model found (cartpole_dqn.pth)")
        print("Train first with: python3.6 train_dqn.py")
        return

    print("Testing DQN model performance...")
    result = test_model_performance("cartpole_dqn.pth", episodes=50)

    if result:
        mean, std, min_val, max_val = result
        print(f"\n✓ Model tested over 50 episodes:")
        print(f"  Mean:   {mean:.1f} ± {std:.1f}")
        print(f"  Range:  {min_val:.0f} - {max_val:.0f}")
        print(f"  Status: {'✓ SOLVED!' if mean >= 195 else '✗ Not solved yet (need 195+)'}")

    # Compare with simple agent
    print("\nComparing with simple baseline...")
    env = cartpole.PyCartPole()
    simple_rewards = []

    for _ in range(50):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Simple heuristic: lean opposite to angle
            action = 1 if state[2] > 0 else 0
            state, reward, done = env.step(action)
            total_reward += reward

        simple_rewards.append(total_reward)

    simple_mean = np.mean(simple_rewards)
    print(f"  Simple Agent: {simple_mean:.1f}")

    # Show comparison
    if result:
        simple_bar_chart(
            [simple_mean, mean, 195.0],
            ["Simple Agent", "DQN Agent", "Solved Threshold"]
        )

        # Show improvement
        improvement = ((mean - simple_mean) / simple_mean) * 100
        if improvement > 0:
            print(f"✓ DQN is {improvement:.1f}% better than baseline")
        else:
            print(f"✗ DQN is {-improvement:.1f}% worse than baseline")

        # Give recommendations
        print("\n" + "="*70)
        print("Recommendations:")
        print("="*70)

        if mean >= 195:
            print("✓ Task solved! Your DQN has mastered CartPole!")
            print("  The agent can balance the pole indefinitely.")
        elif mean >= 150:
            print("⚠ Almost there! Try:")
            print("  - Train for 200 more episodes")
            print("  - The agent is learning well")
        elif mean >= 100:
            print("⚠ Learning but slow. Try:")
            print("  - Train for 500+ episodes")
            print("  - Current learning is on track")
        else:
            print("✗ Not learning effectively. Try:")
            print("  - Retrain from scratch")
            print("  - Check that CUDA is being used")
            print("  - Verify simple agent works (python3.6 train_ai.py)")

        print("="*70 + "\n")


if __name__ == "__main__":
    main()

