# In[]

import collections
import gym
import numpy as np
import math

# In[]

class ValueIterationAgent(object):
    def __init__(self, env, discount = 0.9, fail_reward = 0.0, living_reward = 0.0):
        self.env = env
        self.discount = discount
        self.values = np.zeros((self.env.nS))
        self.policy = np.array([env.action_space.sample() for i in range(self.env.nS)], dtype = int)
        
        # reward and transition matrices
        self.T = np.zeros([self.env.nS, self.env.nA, self.env.nS])
        self.R = np.zeros([self.env.nS, self.env.nA, self.env.nS])
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                transitions = env.P[s][a]
                for p_trans, next_s, reward, done in transitions:
                    self.T[s, a, next_s] += p_trans
                    self.R[s, a, next_s] = reward
                    if done and reward == 0.0:
                        self.R[s, a, next_s] = fail_reward
                    if not done and reward == 0.0:
                        self.R[s, a, next_s] = living_reward

        self.T[s, a, :] /= np.sum(self.T[s, a, :])

    
    def expected_reward(self, s, a):
        value = np.sum(self.T[s,a,:] * (self.R[s,a,:] + self.discount * self.values))
        return value
        
    def learn(self, threshold = 0.0001):
        for it in range(1000000): # Max number of iterations
            delta = 0.
            for s in range(self.env.nS):
                v = self.getValue(s)

                
                new_v = [self.expected_reward(s, a) for a in range(self.env.nA)]
                self.setValue(s, new_v[np.argmax(new_v)])

                delta = max(delta, abs(self.getValue(s) - v))

            #print("{} {} {}".format(v, new_v, delta))
            if delta < threshold:
                break
        
        self.policy = self.extract_policy()
        return

    
    def extract_policy(self):
        
        new_policy = np.zeros_like(self.policy)
        
        for s in range(self.env.nS):
            values = [self.expected_reward(s,a) for a in range(self.env.nA)]
            new_policy[s] = np.argmax(values)
        
        return new_policy
    
        
    def getValue(self, state):
        """
          Return the value of the state.
        """
        return self.values[state]


    def setValue(self, state, value):
        self.values[state] = value


    def act(self, state):
        return self.policy[state]

# In[]
env_name= 'FrozenLake8x8-v0' # 'Taxi-v1'
env = gym.make(env_name)

# In[]

via = ValueIterationAgent(env, 0.5, -0.7, -0.07)
via.learn(0.00001)

# In[]

np.set_printoptions(precision=2, suppress=True)

def v_pp():
   print(np.array(list(via.values)).reshape(env.ncol, env.nrow))


def p_pp():
   mapping = {
       0: '<',
       1: 'v',
       2: '>',
       3: '^'
   }

   print(np.array(list(map(lambda s: mapping[via.policy[s]] if env.desc.flatten()[s] != b'H' else '#',
                       np.arange(env.nS)))).reshape(env.ncol, env.nrow))

v_pp()
p_pp()

# In[]

env.monitor.start('./' + env_name + '-experiment', force=True)

n_episode = 100000
max_time_steps = 1000
total_reward = 0
for i_episode in range(n_episode):

    observation = env.reset() #reset environment to beginning 

    #run for several time-steps
    for t in xrange(max_time_steps): 
        #display experiment
        #env.render() 

        #sample a random action 
        action = via.act(observation)

        #observe next step and get reward 
        observation, reward, done, info = env.step(action)

        if done:
            #env.render()
            total_reward += reward
            print "Simulation finished after {0} timesteps".format(t)
            break

env.monitor.close()
print "Simulation finished with total reward {0}".format(total_reward)

# In[]
gym.upload('./' + env_name + '-experiment', api_key='sk_3MIHM0n0QseEaRQH7mVtxQ', ignore_open_monitors=True)
