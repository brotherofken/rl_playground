# In[]

import collections
import gym
import numpy as np
import math

# In[]

class IterationAgent(object):
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

    def name(self):
        raise("not implemented")
        return
    
    def expected_reward(self, s, a):
        value = np.sum(self.T[s,a,:] * (self.R[s,a,:] + self.discount * self.values))
        return value
        
    def learn(self, threshold = 0.0001):
        raise("not implemented")
        return

    
    def extract_policy(self):
        
        new_policy = np.zeros_like(self.policy)
        
        for s in range(self.env.nS):
            values = [self.expected_reward(s,a) for a in range(self.env.nA)]
            new_policy[s] = np.argmax(values, axis=0)
        
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

class PolicyIterationAgent(IterationAgent):
    def __init__(self, env, discount = 0.9, fail_reward = 0.0, living_reward = 0.0):
        super(PolicyIterationAgent,self).__init__(env, discount, fail_reward, living_reward)

    def name(self):
        return "PolicyIterationAgent"

    def learn(self, threshold = 0.0001):
        while True:
            # Evaluate policy
            for it in range(1000000): # Max number of iterations
                delta = 0.
                for s in range(self.env.nS):
                    v = self.getValue(s)
                    a = self.policy[s]

                    new_v = self.expected_reward(s, a)
                    self.setValue(s, new_v)

                    delta = max(delta, abs(new_v - v))

                print("{} {} {}".format(v, new_v, delta))
                if delta < threshold:
                    break

            # Extract policy
            new_policy = self.extract_policy()
            robust_policy = np.array_equal(new_policy, self.policy)
            self.policy = new_policy
            
            if robust_policy:
                print("Policy is robust now!")
                break
        
        return

# In[]

class ValueIterationAgent(IterationAgent):
    def __init__(self, env, discount = 0.9, fail_reward = 0.0, living_reward = 0.0):
        super(ValueIterationAgent,self).__init__(env, discount, fail_reward, living_reward)

    def name(self):
        return "ValueIterationAgent"
        
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


# In[]
env_name= 'FrozenLake8x8-v0' # 'Taxi-v1'
env = gym.make(env_name)

# In[]

ia = PolicyIterationAgent(env, 1.) #0.5, -0.7, -0.075)
ia.learn(0.00001)

# In[]

ia = ValueIterationAgent(env, 0,99) #0.5, -0.7, -0.075)
ia.learn(0.0000001)

# In[]


if env_name == 'FrozenLake8x8-v0':
    np.set_printoptions(precision=2, suppress=True)
    
    def v_pp():
       print(np.array(list(pia.values)).reshape(env.ncol, env.nrow))
    
    
    def p_pp():
       mapping = {
 
       }
    
       print(np.array(list(map(lambda s: mapping[pia.policy[s]] if env.desc.flatten()[s] != b'H' else '#',
                           np.arange(env.nS)))).reshape(env.ncol, env.nrow))
    
    v_pp()
    p_pp()

# In[]
monitor_name = './' + env_name + '-' + ia.name() + '-experiment'
env.monitor.start(monitor_name, force=True)

n_episode = 1000
max_time_steps = 1000
total_reward = 0
for i_episode in range(n_episode):

    observation = env.reset() #reset environment to beginning 

    #run for several time-steps
    for t in xrange(max_time_steps): 
        #display experiment
        #env.render() 

        #sample a random action 
        action = ia.act(observation)

        #observe next step and get reward 
        observation, reward, done, info = env.step(action)

        if done:
            #env.render()
            total_reward += reward
            print "Simulation finished after {0} timesteps".format(t)
            break

env.monitor.close()
print "Simulation " + monitor_name + " finished with total reward {0}".format(total_reward)

# In[]
gym.upload(monitor_name, algorithm_id='PI_algorithm', api_key='sk_3MIHM0n0QseEaRQH7mVtxQ', ignore_open_monitors=True)
