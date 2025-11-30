#!/usr/bin/env python3
"""
Example Python script to train an AI agent on the CartPole environment.
This demonstrates how to use the Rust CartPole environment from Python.
"""

import numpy as np
import sys
import os

# Try to import the Rust CartPole module (when built with python feature)
try:
    import cartpole
    USING_RUST = True
    print("Using Rust implementation of CartPole")
except ImportError:
    USING_RUST = False
    print("Rust module not found. Using Python fallback.")
    print("To build the Rust module with Python support:")
    print("  cargo build --release --features python")
    print("  Then copy target/release/cartpole.pyd (Windows) or .so (Linux) to this directory")


class SimpleAgent:
    """A simple agent that tries to balance the pole"""

    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        """Simple policy: move cart in direction pole is falling"""
        x, x_dot, theta, theta_dot = state
        # If pole is tilting right (positive theta), push right
        # If pole is tilting left (negative theta), push left
        if theta > 0:
            return 1
        else:
            return 0


def train_simple_agent(episodes=100):
    """Train a simple agent on the CartPole environment"""
    if not USING_RUST:
        print("\nCannot run training without Rust module.")
        return

    # Create environment
    env = cartpole.PyCartPole()
    agent = SimpleAgent(env)

    print(f"\n=== Training Simple Agent for {episodes} episodes ===\n")

    total_rewards = []

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

        total_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1:3d} | Steps: {steps:3d} | Avg Reward (last 10): {avg_reward:.1f}")

    print(f"\n=== Training Complete ===")
    print(f"Average reward over all episodes: {np.mean(total_rewards):.1f}")
    print(f"Best episode: {max(total_rewards):.1f}")
    print(f"Worst episode: {min(total_rewards):.1f}")


def test_environment():
    """Test the CartPole environment"""
    if not USING_RUST:
        print("\nCannot test without Rust module.")
        return

    print("\n=== Testing CartPole Environment ===\n")

    env = cartpole.PyCartPole()

    # Print environment info
    obs_space = env.observation_space()
    print(f"Observation space: Low={obs_space[0]}, High={obs_space[1]}")
    print(f"Action space: {env.action_space()} discrete actions (0=left, 1=right)")

    # Run a random episode
    print("\n--- Running random episode ---")
    state = env.reset()
    print(f"Initial state: {state}")

    done = False
    steps = 0
    total_reward = 0

    while not done and steps < 100:
        action = np.random.randint(0, 2)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 10 == 0:
            print(f"Step {steps}: action={action}, state={state}, reward={reward}")

    print(f"\nEpisode finished after {steps} steps with total reward {total_reward}")


if __name__ == "__main__":
    if USING_RUST:
        test_environment()
        train_simple_agent(episodes=50)
    else:
        print("\n" + "="*60)
        print("To use this script, you need to build the Rust module:")
        print("="*60)
        print("\nOn Windows:")
        print("  1. Ensure Python is in your PATH")
        print("  2. Run: cargo build --release --features python")
        print("  3. Copy target/release/cartpole.pyd to this directory")
        print("\nOn Linux (including Jetson Nano):")
        print("  1. Ensure Python development headers are installed:")
        print("     sudo apt-get install python3-dev")
        print("  2. Run: cargo build --release --features python")
        print("  3. Copy target/release/libcartpole.so to cartpole.so")
        print("\nThen run this script again!")
        print("="*60)

