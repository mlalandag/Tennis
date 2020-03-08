[//]: # (Image References)

# Project 3: Collaboration and Competition

### Introduction
 
![Plot](Tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the root folder of the repo and decompress it. 

## Installation

1. Download or clone this GitHub repository.

2. Download and install Anaconda Python 3.6 from the following link : https://www.anaconda.com/download/

2. Create a new conda environment named drlnd (or whatever name you prefer) and then activate it:

	- Linux or Mac:
	
		`conda create --name drlnd python=3.6`
	
		`source activate drnld`

	- Windows:
	
		`conda create --name drnld python=3.6`
	
		`activate drnld`

4. Install the required dependencies navigating to where you downloaded and saved this GitHub repository and then into the '.python/' subdirectory. Then run from the command line:
	
		`pip3 install .`
		
5. To enable the use of GPUs in the conda environment install

    `conda install pytorch torchvision cuda90 -c pytorch`

    Check gpu install

    `python -c 'import torch; print(torch.rand(2,3).cuda())'`
 
## Files

- agent.py: Contains the agent who interacts with the environment and is used to train the model. 
- multi_agent: Instantiates the two agents and a shared ReplayBuffer and implement the step and act methods. 
- model.py: Contains the Neural Networks implemented in Pytorch that is used to pick the actions (actor) and evaluate them (Critic). 
- replay_buffer.py: Helper class to implement the Esperience Replay algorithm.
- OUNoise.py: Helper Class to implement the addition of some noise to the actions chosen by the agents
- agents_training.py: Process that delivers the trained model. 
- agents_test.py: Execution of some episodes with the agent using the trained model. 

## Training

 - Go to the root folder of the repo and run:
 
 	`python agents_training.py`
	
 - When the score reaches the value 0.5 it will stop and save the model weights to the file .

## Testing

 - To test the trained agent:
 
 	`python agents_test.py`
 	
### The base DDPG algorithm for this project is taken from the following udacity repo:

https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal	
 
