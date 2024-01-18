# train.py

import numpy as np
import torch
from env import AtariEnv  # Assuming AtariEnv class is in env.py
from agent import Agent  # Assuming Agent class is in agent.py
from model import DeepQNetwork  # Assuming DeepQNetwork is in model.py
from collections import deque

if __name__ == '__main__':
    # Set the parameters for the training
    n_episodes = 1000  # Number of episodes to train
    render = False     # Set to True if you want to render the environment on screen

    # Environment setup
    env = AtariEnv('PongNoFrameskip-v4')
    input_dims = (env.observation_space.shape[0], env.observation_space.shape[1], 4)  # Adjust as per your env
    n_actions = env.action_space.n

    # Agent setup
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.00025, input_dims=input_dims, batch_size=32, n_actions=n_actions, 
                  max_mem_size=100000, eps_end=0.01, eps_dec=1e-5)

    scores, eps_history = [], []
    for i in range(n_episodes):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

            if render:
                env.render()

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(f'Episode {i}: Score {score}, Average Score {avg_score}, Epsilon {agent.epsilon}')

        # Optional: save the model and other metrics every N episodes

    # Close the environment
    env.close()

    # Optional: plot the training results
