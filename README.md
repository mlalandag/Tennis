[//]: # (Image References)

# Project 2: Continuous Control

### Introduction
 
![Plot](Reacher.png)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment

In this particular case i´m solving the first versión of the environment, which is the one with only one agent. The goal is to 
 get an average score of +30 over 100 consecutive episodes. The algorithm chosen to solve the task is DDPG.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
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
- model.py: Contains the Neural Networks implemented in Pytorch that is used to pick the actions and evaluate them. 
- replay_buffer.py: Helper class to implement the Esperience Replay algorithm.
- OUNoise.py: Helper Class to implement the addition of some noise to the actions chosen by the agent 
- agent_training.py: Process that delivers the trained model. 
- agent_testing.py: Execution of some episodes with the agent using the trained model. 

## Training

 - Go to the root folder of the repo and run:
 
 	`python agent_training.py`
	
 - When the score reaches the value +13 it will stop and save the model weights to the file .

## Testing

 - To test the trained agent:
 
 	`python agent_test.py`
 	
### The base algorithm for this project is taken from the following udacity repo:

https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal	
 
	
