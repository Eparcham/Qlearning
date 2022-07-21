import gym
import numpy as np
import random
import matplotlib.pyplot as plt

## makes environment deterministic
from gym.envs.registration import register
register(
   id='FrozenLake-v1',
   entry_point='gym.envs.toy_text:FrozenLakeEnv',
   kwargs={'map_name' : '8x8', 'is_slippery': True},
   max_episode_steps=200,
   reward_threshold=0.78, # optimum = .8196
)
env = gym.make('FrozenLake-v1')
# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter

learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.00001             # Exponential decay rate for exploration prob
# Plotting Metrix
reward_list = []
reward_ = []
Prob_F = []
episode_number = 100000
reward_count = 0
for i in range(1,episode_number):
    
    state = env.reset()
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)

    while True:
              
        # exploit vs explore to find action
        # %05 = explore, %95 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # action process and take reward/ observation
        next_state, reward, done, prob = env.step(action)
        
        # Q learning function
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

        # update state
        state = next_state
        
        reward_count += reward
        reward_.append(reward)
        
        if done:
            Prob_F.append(prob['prob'])
            break
                
    if i%10 == 0:
        reward_list.append(reward_count)
        print("Episode: {}, total reward {} average reward {} epsilon {}".format(i,reward_count,reward_count/i,epsilon))
           
plt.plot(reward_list)
plt.show()
print("*"*100,"\n")
print("Q Value: {} \n".format(q_table))