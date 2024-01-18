"""
    Answer the following questions
•	What algorithm? 
    o	Deep Q Network, a model free RL algorithm.
    o	Learns the value in a particular state.
    o	Specific variant:  involves experience replay and a deep CNN
•	What sort of data structures and classes will we need?
    o	Input 
        	Take maximum value for each pixel color value over the frame being encoded and the previous frame.
        	Extract Y channel, that is Luminance, from RGB frame to rescale to 84 x 84.
    o	DQN class , for Deep Q network
    o	ReplayMemory class, memory representation for agent
    o	Agent Class, performs, remembers and learns actions
    o	Create a separate py file for environment
    o	Create separate py file for settings
•	What model architecture?
    o	architecture
        	input:  84 x 84 x 4 image produced by pre-processing map psi
        	First hidden layer – convolutional, 32 filters of 8 x 8, with stride 4.  Rectifier nonlinearity (ReLU)
        	Second hidden layer – convolutional, 64 filters of 4 x 4, stride 2.  Rectifier nonlinearity.
        	Third hidden layer – convolutional, 64 filters of 3 x 3.  Stride 1.  Rectifier nonlinearity. 
        	Final hidden layer – fully connected.  512 rectifier units.
        	Output layer - fully connected.  Single output for each valid action.  Valid actions varied between 4 and 18.
o	Training Details
    	49 Atari games.
    	A different network for each game.  However, the architecture was not changed.
    	Reward clipping.
        •	Positive rewards clipped at 1.
        •	Negative rewards clipped at -1.
        •	0 rewards unchanged.
    	If there is a live counter, Atari emulator sends the number of lives at the end of the game.  This signal was used to mark the end of the episode during training.
•	What are the hyper parameters
    o	Mini batch size 32
    o	Replay memory size 1,000,000
        	50,000 frames takes up about 17GB or RAM.  Scale accordingly.  I have 32 GB, but I only want to use up to 16 GB
    o	Agent history length 4
    o	Target network update frequency 10,000
    o	Discount factor 0.99
    o	Action repeat 4
    o	Update frequency 4
    o	Learning rate 0.00025
    o	Gradient momentum 0.95
    o	squared gradient momentum 0.95
    o	min squared gradient 0.01
    o	initial exploration 1
    o	final exploration 1,000,000
    o	no-op max 30

"""

import gym
import numpy as np
import torch as T
from deep_q_learning import Agent
from utils import plot_learning_curve, make_env
from gym import wrappers

class AtariEnv():
    def __init__(self, game_name, n_games=1, no_ops=30, agent_history_length=4, frame_skip=4):
        self.env = make_env(game_name)
        self.n_games = n_games
        self.no_ops = no_ops
        self.agent_history_length = agent_history_length
        self.frame_skip = frame_skip
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        self.state = None
        self.last_lives = 0
        
    def reset(self):
        observation = self.env.reset()
        self.last_lives = 0
        terminal_life_lost = True
        for _ in range(np.random.randint(self.no_ops)):
            observation, _, _, _ = self.env.step(1)
        
        processed_observation = self.process_frame(observation)
        self.state = np.repeat(processed_observation, self.agent_history_length, axis=0)
        
        return terminal_life_lost
    
    def step(self, action):
        new_observation, reward, terminal, info = self.env.step(action)
        
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        
        processed_new_observation = self.process_frame(new_observation)
        new_state = np.append(self.state[1:, :, :], processed_new_observation, axis=0)
        
        return new_state, reward, terminal_life_lost, info
    
    def render(self):
        self.env.render()
        
    def close(self):
        self.env.close()
        
    def sample_action(self):
        return self.env.action_space.sample()
    
    def process_frame(self, frame):
        frame = np.mean(frame, axis=2).astype(np.uint8)
        frame = frame[34:34+160, :160]
        frame = frame[::2, ::2]
        return frame
    
if __name__ == '__main__':
    env = AtariEnv('PongNoFrameskip-v4', n_games=1)
    n_games = 10000
    scores = []
    eps_history = []
    
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.agent_history_length, env.observation_space.shape[0], env.observation_space.shape[1]), n_actions=env.action_space.n, mem_size=50000, eps_min=0.1, batch_size=32, replace=1000, eps_dec=1e-5, chkpt_dir='models/', algo='DQNAgent', env_name='PongNoFrameskip-v4')
    
    for i in range(n_games):
        done = False
        score = 0
        observation_tuple = env.reset()
        observation = observation_tuple[0]
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_tuple[0]
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' % \
                (score, avg_score, agent.epsilon))
        
        if i % 10 == 0 and i > 0:
            agent.save_models()
            
    filename = 'pong_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
    

