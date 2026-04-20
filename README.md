
# Keras Q-Learning for CartPole

## Overview

This project implements a simple Deep Q-Learning agent using Keras and Gymnasium.

The agent is trained on the `CartPole-v1` environment, where the goal is to balance a pole on a moving cart by choosing one of two actions: move left or move right.

The notebook uses a neural network to approximate Q-values instead of storing a Q-table. This makes it a basic Deep Q-Network style implementation.

## What the Project Does

1. Installs and imports Gymnasium.
2. Checks the versions of NumPy and TensorFlow.
3. Creates the `CartPole-v1` environment.
4. Sets random seeds for reproducibility.
5. Builds a neural network to approximate Q-values.
6. Uses epsilon-greedy action selection.
7. Stores experiences in replay memory.
8. Samples random mini-batches from memory.
9. Updates Q-value targets using the Bellman equation.
10. Trains the model through experience replay.
11. Evaluates the trained agent over 10 episodes.

## Environment

The project uses:

- `CartPole-v1`

The state has 4 values:

- cart position
- cart velocity
- pole angle
- pole angular velocity

The action space has 2 actions:

- move cart left
- move cart right

The goal is to keep the pole balanced for as many time steps as possible.

## Model Structure

The Q-learning model is a small fully connected neural network.

Architecture:

- `Input(shape=(state_size,))`
- `Dense(24, activation='relu')`
- `Dense(24, activation='relu')`
- `Dense(action_size, activation='linear')`

For CartPole:

- `state_size = 4`
- `action_size = 2`

So the model takes the 4-dimensional environment state as input and outputs 2 Q-values, one for each possible action.

The final layer uses a linear activation because Q-values are continuous numeric estimates, not class probabilities.

## Q-Learning Logic

### Epsilon-Greedy Action Selection

The notebook uses epsilon-greedy exploration:

- with probability `epsilon`, the agent chooses a random action
- otherwise, it chooses the action with the highest predicted Q-value

Initial values:

- `epsilon = 1.0`
- `epsilon_min = 0.01`
- `epsilon_decay = 0.99`

At the beginning, the agent explores heavily. As training continues, epsilon decreases, so the agent gradually relies more on the learned Q-values.

### Replay Memory

The project uses replay memory with:

- `deque(maxlen=2000)`

Each stored experience contains:

- current state
- action taken
- reward received
- next state
- whether the episode ended

Replay memory helps the model train on randomized past experiences instead of only the most recent transition.

### Bellman Update

For each sampled experience, the target Q-value is updated using:

- immediate reward if the episode is done
- reward plus discounted future reward if the episode is not done

The discount factor used is:

- `gamma = 0.95`

So the model learns to estimate long-term reward, not just immediate reward.

## Training

The notebook trains the agent for:

- `episodes = 10`
- maximum `200` steps per episode
- replay batch size `64`
- training every `5` steps

During each episode, the agent:

1. observes the current state
2. chooses an action
3. steps through the environment
4. stores the experience
5. trains from replay memory every few steps
6. reduces epsilon over time

## Why the Optimizer and Loss Function Were Chosen

### Optimizer: Adam

The model uses the Adam optimizer with:

- `learning_rate = 0.001`

Adam is used because it is a practical default optimizer for neural networks. It adapts learning rates during training and usually works more smoothly than plain stochastic gradient descent.

For this project, Adam helps update the Q-network weights efficiently while learning from replay-memory batches.

### Loss Function: Mean Squared Error

The model uses:

- `loss='mse'`

Mean squared error is used because the model is doing Q-value regression.

The network predicts continuous Q-values, and training tries to make the predicted Q-value closer to the Bellman target value. Since this is a numeric prediction problem, MSE is an appropriate loss.

## Evaluation

After training, the notebook evaluates the agent for 10 episodes.

Evaluation scores were:

- episode 1: score 9
- episode 2: score 8
- episode 3: score 9
- episode 4: score 8
- episode 5: score 8
- episode 6: score 9
- episode 7: score 9
- episode 8: score 9
- episode 9: score 9
- episode 10: score 9

## Result Analysis

The final result is weak.

The trained agent only survives for around 8 to 9 steps during evaluation. For `CartPole-v1`, a strong agent should survive for hundreds of steps, with the maximum episode length being 500.

This means the model did not learn a useful control policy yet.

Likely reasons:

- only 10 training episodes were used
- replay memory did not have enough experience
- epsilon was still high during much of training
- the network had very little time to improve
- no target network was used
- no Double DQN or stabilization method was used
- the evaluation scores are close to random behavior

So the notebook correctly demonstrates the structure of Deep Q-Learning, but the trained agent is not successful yet.

## Technical Characteristics

- reinforcement learning with Gymnasium
- Deep Q-Learning style neural Q-value approximation
- epsilon-greedy exploration
- replay memory with `deque`
- mini-batch replay training
- Bellman target updates
- Keras Sequential model
- linear output layer for Q-values
- CartPole control environment

## Packages Used

- `gymnasium`
- `tensorflow`
- `numpy`
- `random`
- `collections`
- `os`
- `sys`
- `warnings`

Keras components used:

- `tensorflow.keras.models.Sequential`
- `tensorflow.keras.layers.Input`
- `tensorflow.keras.layers.Dense`
- `tensorflow.keras.optimizers.Adam`

## Files

- `Keras-Qlearning.ipynb`
- `README.md`

## Summary

This project demonstrates a simple Deep Q-Learning workflow in Keras using the CartPole environment. It builds a neural network to predict Q-values, uses epsilon-greedy action selection, stores experiences in replay memory, trains using Bellman targets, and evaluates the trained agent. The implementation shows the main technical structure of Deep Q-Learning, but the current training result is weak because the agent only survives for about 8 to 9 steps during evaluation.

