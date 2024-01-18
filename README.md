# atari-games-q-network

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

I will use PongNoFrameskip-v4' to solve this problem.  I do not have the GPU's to solve multiple games.

This is based on the paper "Human-level control through deep reinforcement
learning."

As before I placed the following contraint upon myself, that is that I refused to use frameworks or API's such as HuggingFace, as an example.  The point is this model is to help me understand RL and Deep Q Networks.  Using those API's would not help my understanding.  
