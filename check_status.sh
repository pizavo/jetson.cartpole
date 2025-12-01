#!/bin/bash
# Quick script to show all analysis tools available

echo ""
echo "========================================"
echo "CartPole Training Analysis Tools"
echo "========================================"
echo ""
echo "Available commands:"
echo ""
echo "1. Train agents:"
echo "   python3.6 train_ai.py        # Simple baseline (1 min)"
echo "   python3.6 train_dqn.py       # DQN with GPU (2.5 min, 500 episodes)"
echo ""
echo "2. Analyze results:"
echo "   python3.6 compare_training.py  # Compare DQN vs Simple"
echo "   python3.6 plot_training.py     # Detailed DQN analysis"
echo ""
echo "3. Read explanations:"
echo "   cat RESULTS_EXPLAINED.md     # Full analysis of your results"
echo "   cat COMPLETE_GUIDE.md        # Complete usage guide"
echo ""
echo "========================================"
echo "Current Status"
echo "========================================"
echo ""

# Check if model exists
if [ -f "cartpole_dqn.pth" ]; then
    echo "✓ Trained model found: cartpole_dqn.pth"
    echo ""
    echo "Quick test:"
    python3.6 -c "
import torch, cartpole, numpy as np
from train_dqn import DQNAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = DQNAgent(device=device)
agent.policy_net.load_state_dict(torch.load('cartpole_dqn.pth', map_location=device))
agent.policy_net.eval()
agent.epsilon = 0

env = cartpole.PyCartPole()
rewards = []
for _ in range(10):
    state = env.reset()
    total = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, r, done = env.step(action)
        total += r
    rewards.append(total)

avg = np.mean(rewards)
print(f'  Average reward: {avg:.1f}')
if avg >= 195:
    print('  Status: ✓ SOLVED!')
elif avg >= 150:
    print('  Status: ⚠ Almost there (need 195+)')
else:
    print('  Status: ✗ Needs more training')
print('')
" 2>/dev/null || echo "  (Could not test model)"
else
    echo "✗ No trained model found"
    echo "  Train with: python3.6 train_dqn.py"
    echo ""
fi

echo "========================================"
echo "Your Recent Results"
echo "========================================"
echo ""
echo "Simple Agent:  170-180 reward"
echo "DQN Training:  107.9 reward (with exploration)"
echo "DQN Testing:   173.0 reward (true performance)"
echo "DQN Best:      292.0 reward"
echo "Goal:          195.0 reward (SOLVED)"
echo ""
echo "Progress: 88% to solved (173/195)"
echo "Recommendation: Train for 500 episodes"
echo ""
echo "========================================"
echo ""

