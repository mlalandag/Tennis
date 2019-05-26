
from unityagents import UnityEnvironment
import numpy as np
import torch
import time
from agent import Agent

num_agents = 2

# please do not modify the line below
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


max_num_episodes = 5

agent = Agent(state_size, action_size, 1)

agent.actor_local.load_state_dict(torch.load('actor_weights.pth'))
agent.critic_local.load_state_dict(torch.load('critic_weights.pth'))

for episode in range(1, max_num_episodes+1):
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment 
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    while True:
        
        action = agent.act(state)                       # select an action (for each agent)
        action = np.clip(action, -1, 1)                 # all actions between -1 and 1
        env_info = env.step(action)[brain_name]         # send all actions to tne environment
        next_state = env_info.vector_observations[0]    # get next state (for each agent)
        reward = env_info.rewards[0]                    # get reward (for each agent)
        done = env_info.local_done[0]                   # see if episode finished
        score += reward                                 # update the score
        state = next_state                              # roll over the state to next time step
        time.sleep(0.005)
        if np.any(done):                                # exit loop if episode finished
            break
    
    print('\rEpisode {}\tScore: {:.2f}'.format(episode, score))
     
        
#When finished, you can close the environment.
env.close()
