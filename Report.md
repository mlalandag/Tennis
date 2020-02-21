[//]: # (Image References)

# Report

## Learning Algorithm

In order to solve the Reacher environment for this project we implement the DDPG (deep Deterministic Policy Gradient) 
algorithm. This algorithm combines the advantages of value-based methods and policy-based methods such as and uses 
four neural networks: a Q network (the Critic), a policy network (the Actor), a target Q network, and a target policy 
network which are described in the Model section. 

The actor network maps states to actions and delivers directly the action to take (not a probability distribution), 
this is why this kind of algorithms (policy based) are good for environments with a continuous action space.

The Critic evaluates the quality of actions taken by the Actor (if it is good or not) and speed up learning.

Experience replay is used to obtain samples of the environment past states, actions, rewards and next states tuples 
that are not correlated with each other. Experience replay (implemented in the ReplayBufferbuffer.py file and used 
by the agent_step method) consists in keeping track of the tuples (state, action,reward, next state, done)
encountered by the agent in a buffer and, after a number of steps, sample from it some of them in a way that they 
are not correlated. These samples are then used to optimize the neural networks in the agent´s "learn" method.

In these kind of problems, with continuous action spaces exploration is done via adding noise to the action itself and
not selecting ramdom actions. We use the Ornstein-Uhlenbeck process implemented the OUNoise.py file to obtain the 
precise noise to add to the action.

Another Technique used by the algorithm is the soft update of the target weights, which after each step taken slowly
approximates the target neural network´s weights to the actual actor and critic network´s weights.
  
### Model

* **Actor**

* The Neural Network has two layers Hidden layers with 256 neurons each.
* The activation function used is `ReLU` for the input and first layer, and `tanh` for the output
* The output layer has 4 values which corresponds to the dimension of each action
* Batch Normalization is used on the inputs of each layer

* **Critic**

* The Neural Network has two layers Hidden layers with 256 neurons each.
* The activation function used is `ReLU` for the input and first layer, and none for the output
* The output layer has just 1 value which corresponds to the assesment made by the critic of the action chosen 
by the actor
* Batch Normalization is used on the inputs of each layer

### Chosen hyperparameters

* BUFFER_SIZE = int(1e5)    (replay buffer size)
* BATCH_SIZE = 128          (minibatch size)
* GAMMA = 0.99              (discount factor)
* TAU = 1e-3                (for soft update of target parameters)
* LR_ACTOR = 1e-5           (learning rate of the actor)
* LR_CRITIC = 1e-4          (learning rate of the critic)
* WEIGHT_DECAY = 0          (L2 weight decay)

### Plot of Rewards

![Plot](Scores.png)

### Ideas for Future Work

In order to improve the efficiency of the agent order algorithms can be tried. Those are:

* **Prioritized experience replay**
* **PPO**
* **D4PG**

