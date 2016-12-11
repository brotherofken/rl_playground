# In[]

import collections
import gym
import numpy as np
import math


# In[]
class DiscretePolicy(object):
    def __init__(self, env):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env
        self.policy = np.array([env.action_space.sample() for i in
                                range(self.env.nS)],
                               dtype=int)

    def action(self, state):
        return self.policy[state]


# In[] Monte Carlo Discrete Model Free Predictor
class MonteCarloDMFPredictor(object):
    def __init__(self, env):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env

    def evaluate(self, policy, iterations=1000, discount=1.,
                 every_visit=True):
        if not isinstance(policy, DiscretePolicy):
            raise Exception('policy should have type DiscretePolicy')

        counts = np.zeros((self.env.nS))
        values = np.zeros((self.env.nS))

        max_time_steps = 500
        for i_episode in xrange(iterations):
            # reset environment to beginning
            observations = np.zeros((max_time_steps), dtype=np.int)
            rewards = np.zeros((max_time_steps))

            # generate an episode using policy
            observations[0] = env.reset()
            steps = 0
            for t in xrange(1, max_time_steps):
                # sample a random action
                action = policy.action(observations[t - 1])
                # observe next step and get reward
                observations[t], rewards[t - 1], done, info = env.step(action)
                #observations[t] = observation
                if done:
                    steps = t + 1
                    break
            if steps <= 1:
                continue

            if i_episode % 1000 == 0:
                print 'Episode {} finished in {} steps.'.format(i_episode,
                                                                steps)

            observations = observations[:steps]
            rewards = rewards[:steps]

            returns = np.zeros((steps))
            returns[-1] = rewards[-1]
            for t in reversed(xrange(steps - 1)):
                returns[t] = discount * returns[t + 1] + rewards[t]

            visited = np.zeros((self.env.nS), dtype=np.bool)
            for t in xrange(steps):
                s = observations[t]
                if every_visit or not visited[s]:
                    counts[s] += 1.
                    values[s] += (returns[t] - values[s]) / counts[s]
                    visited[s] = True
        return values


# In[] TD(0) Discrete Model Free Predictor
class TD0DMFPredictor(object):
    def __init__(self, env, alpha=0.01):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env
        self.alpha = alpha

    def evaluate(self, policy, iterations=1000, discount=1.,
                 every_visit=True):
        if not isinstance(policy, DiscretePolicy):
            raise Exception('policy should have type DiscretePolicy')

        values = np.zeros((self.env.nS))

        max_time_steps = 500
        for i_episode in xrange(iterations):
            # generate an episode using policy
            s = env.reset()
            for t in xrange(1, max_time_steps):
                # sample a random action
                action = policy.action(s)
                # observe next step and get reward
                sp, r, done, info = env.step(action)
                td_target = r + discount * values[sp]
                values[s] += self.alpha * (td_target - values[s])
                if done:
                    if t % 1000 == 0:
                        msg = 'Episode {} finished in {} steps.'
                        print msg.format(i_episode, t)
                    break
                s = sp

        return values


# In[] TD(l) Discrete Model Free Predictor
class TDlDMFPredictor(object):
    def __init__(self, env, alpha=0.01, llambda=0.9):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env
        self.alpha = alpha
        self.llambda = llambda

    def evaluate(self, policy, iterations=1000, discount=1.,
                 every_visit=True):
        if not isinstance(policy, DiscretePolicy):
            raise Exception('policy should have type DiscretePolicy')

        values = np.zeros((self.env.nS))
        e = np.zeros((self.env.nS))

        max_time_steps = 500
        for i_episode in xrange(iterations):
            # generate an episode using policy
            s = env.reset()
            for t in xrange(1, max_time_steps):
                # sample a random action
                action = policy.action(s)
                # observe next step and get reward
                sp, r, done, info = env.step(action)

                td_error = r + discount * values[sp] - values[s]
                e[s] += 1

                for ss in xrange(self.env.nS):
                    values[ss] += self.alpha * td_error * e[ss]
                    e[ss] = self.alpha * self.llambda * e[ss]

                if done:
                    if i_episode % 1000 == 0:
                        msg = 'Episode {} finished in {} steps.'
                        print msg.format(i_episode, t)
                    break
                s = sp

        return values

if False:
    # In[]
    env_name = 'FrozenLake8x8-v0'  # 'Taxi-v1'
    env = gym.make(env_name)
    
    # In[]
    np.random.seed(43)
    policy = DiscretePolicy(env)
    
    policy.policy = ia.policy
    
    mfp = MonteCarloDMFPredictor(env)
    mfp = TD0DMFPredictor(env, alpha=0.001)
    mfp = TDlDMFPredictor(env, alpha=0.001, llambda=0.9)
    
    values = mfp.evaluate(policy, iterations=10000, discount=0.9)
    
    print values.reshape((8, 8))
    print
    print ia.values.reshape((8, 8))
    
    # In[]

