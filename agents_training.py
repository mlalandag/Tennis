
from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from multi_agent import MultiAgent

num_agents = 2

# please do not modify the line below
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('States: ', states)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


max_num_episodes = 5000

episode_scores = []

agents = MultiAgent(num_agents, state_size, action_size, 1)

print("loop over episodes")

for episode in range(1, max_num_episodes+1):
     
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    agents.reset()    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)

    while True:

        actions = agents.act(states)                       # select an action
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished

        #actions = np.squeeze(actions, axis=1)

        agents.step(states, actions, rewards, next_states, dones)

        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    episode_scores.append(np.max(scores))

    mean_last_100 = np.mean(episode_scores[episode - 100:])

    print('Total score for episode {} : {:.2f} - Mean last 100 episodes {:.2f}'.format(episode, np.max(scores), mean_last_100))

    if mean_last_100 > 0.5:
        
        for id, agent in enumerate(agents.agents):
                torch.save(agent.actor_local.state_dict(),  'weights_agent_' + str(id) + '_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'weights_agent_' + str(id) + '_critic.pth')

        break

#Plot scores and save to image file
graph = plt.figure()
plt.plot([score for score in range(len(episode_scores))], episode_scores)
plt.ylabel('scores')
plt.xlabel('episodes')
plt.show()
graph.savefig('scores.jpg')

#When finished, you can close the environment.
env.close()
        
