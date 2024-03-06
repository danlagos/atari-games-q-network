# atari-games-q-network
## PROJECT NOT COMPLETE.  

You must use python 3.6 to run this code.

This is my second Deep Q Learning network.  The first DQN model I built had instability problems for the following reasons:

* One network was used to for actions and for the value of the actions.
* This created correlations.
* Small updates to Q may have been significantly changing the policy.

As a solution I propose the following:

* Create a class to handle the experience replay.
    * Experience replay is the state at time t.
    * The action the agent will take in that time step.
    * The reward the agent receives.
    * The observation after the action taken.
* A separate deep neural network to handle the calculation of the target values.
    * Update neural network with experience replay.
    * Sample those memories uniformly.
    * Regarding correlations:  uniform sampling guarantees we do not get a string of observations from a single episode and thereby break the correlations.
* Why these solutions:  
    * The point is to solve the Bellman Equation.
    * To decouple the correlations between the action and the value of the actions
    * By using a class for experience replay we are randomizing over the data to remove the correlations and smoothing over changes in the data distribution.

I will use PongNoFrameskip-v4 to solve this problem.  I do not have the GPU's to solve multiple games.

This is based on the paper "Human-level control through deep reinforcement
learning."  I did not use any code from the paper.  From the paper I build pseudo code that answers the following questions:

* What algorithm?
* What sort of date structures and classes will I need?
* What model architecture?
* What are the hyperparameters?
* what general results can I expect?

Pseudo code found in pdf file simply titled "pseudo_code."  This is not in a format that is used to write pseudo code found in peer reviewed articles.  This is meant to show what I extracted from the article this model is based on.  It is also the initial plan I used to write the code.

I had to make modifications to the algorithm because of my lack of GPU's.  For example, instead of max_mem_size = 1,000,000 I used max_mem_size = 100,000.  I may further reduce this upon final implementation given my GPU and RAM.

As before I placed the following constraint upon myself:  I refused to use frameworks or API's such as HuggingFace.  The point is to help me understand RL and Deep Q Networks.  Using those API's would not help my understanding.  Buidling networks from scratch will.

| Project Description | Skill |
|----------------------|-----------------------------------------|
| The project involves building a second Deep Q Learning (DQN) network to address instability issues identified in the first model. | Deep Learning |
| Identified problems include the use of a single network for both actions and value estimations, creating correlations and policy instability. | Problem Identification in AI Models |
| Proposed solutions include creating a class for experience replay, using a separate deep neural network for target value calculations, and updating the neural network with uniformly sampled experience replays to reduce correlations. | Solution Design and Implementation in Machine Learning |
| The model aims to solve the Bellman Equation, decouple action-value correlations, and ensure data randomization to smooth data distribution changes. | Mathematical Modeling in Machine Learning |
| The PongNoFrameskip-v4 game is used for testing the solution, constrained by available GPU resources. | Practical Application of DQN in Gaming |
| Based on the "Human-level control through deep reinforcement learning" paper, the project did not use the paper's code but developed pseudo code to outline algorithms, data structures, model architecture, hyperparameters, and expected results. | Research and Development in Deep Learning |
| Adjustments were made to the original algorithm to accommodate limited GPU resources, such as reducing the maximum memory size for experience replay. | Adaptation and Optimization for Resource Constraints |
| The project deliberately avoids using frameworks like HuggingFace to deepen understanding of RL and DQN by building networks from scratch. | Deep Learning Frameworks and Libraries Independence |
