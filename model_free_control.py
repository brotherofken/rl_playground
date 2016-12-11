# In[]
import gym
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# In[]

def p_pp(policy):
   mapping = {
      0: '<',
      1: 'v',
      2: '>',
      3: '^'
   }
   lpolicy = np.argmax(policy, axis=1).ravel()
   print(np.array(list(map(lambda s: mapping[lpolicy[s]] if env.desc.flatten()[s] != b'H' else '#',
                           np.arange(env.nS)))).reshape(env.ncol, env.nrow))

# In[]
class StochasticDiscretePolicy(object):
    def __init__(self, env):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env
        self.policy = np.random.rand(self.env.nS, self.env.nA)
        row_sums = self.policy.sum(axis=1).reshape(self.env.nS, 1)
        self.policy = self.policy / row_sums

    def action(self, state):
        return np.random.choice(range(self.env.nA),
                                p=self.policy[state, :].ravel())

    def e_greedy_update(self, qvalues, eps):
        for s in range(self.env.nS):
            actions = values[s, :].ravel()
            a_star = np.argwhere(actions == np.amax(actions))
            a_star = np.random.choice(a_star.ravel())

            policy.policy[s, :] = 0.0
            policy.policy[s, a_star] = 1.0

# In[] Monte Carlo Discrete Model Free Predictor
class MonteCarloDMFController(object):
    def __init__(self, env):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env

    def _generate_episode(self):
        max_time_steps = 500
        # reset environment to beginning
        states = np.zeros((max_time_steps), dtype=np.int)
        actions = np.zeros((max_time_steps), dtype=np.int)
        rewards = np.zeros((max_time_steps))

        # generate an episode using policy
        states[0] = env.reset()
        steps = 0
        for t in xrange(1, max_time_steps):
            # sample a random action
            actions[t - 1] = policy.action(states[t - 1])

            # observe next step and get reward
            states[t], rewards[t - 1], done, info = env.step(actions[t - 1])
            # observations[t] = observation
            if done:
                steps = t # + 1
                break

        # for array in (states, actions, rewards):
        #     array = array[:steps]
        states = states[:steps]
        actions = actions[:steps]
        rewards = rewards[:steps]
        return states, actions, rewards, steps

    def _calculate_returns(self, rewards, steps, discount):
        returns = np.zeros((steps))
        returns[-1] = rewards[-1]
        for t in reversed(xrange(steps - 1)):
            returns[t] = discount * returns[t + 1] + rewards[t]
        return returns

    def learn(self, policy, iterations=1000, discount=1., eps=0.9,
              every_visit=False):

        if not isinstance(policy, StochasticDiscretePolicy):
            raise Exception('policy should have type DiscretePolicy')

        counts = np.zeros((self.env.nS))
        values = np.zeros((self.env.nS, self.env.nA))

        for i_episode in xrange(iterations):
            # a. generate episode
            states, actions, rewards, steps = self._generate_episode()

            if steps <= 1:
                continue
            if i_episode % 1000 == 0:
                print 'Episode {} finished in {} steps.'.format(i_episode,
                                                                steps)
                p_pp(policy)

            # b. Calculate returns
            returns = self._calculate_returns(rewards, steps, discount)

            # c. Update action-statevalues
            visited = np.zeros((self.env.nS), dtype=np.bool)
            for t in xrange(steps):
                s, a, r = states[t], actions[t], returns[t]
                if every_visit or not visited[s]:
                    counts[s] += 1.
                    values[s, a] += (r - values[s, a]) / counts[s]
                    visited[s] = True

            # d. update policy
            policy.e_greedy_update(values, eps)
#            for s in range(self.env.nS):
#                # random of argmax actions
#                actions = values[s, :].ravel()
#                a_star = np.argwhere(actions == np.amax(actions))
#                a_star = np.random.choice(a_star.ravel(), 1)[0]
#
#                for a in range(self.env.nA):
#                    if a == a_star:
#                        policy.policy[s, a] = 1 - eps + eps / self.env.nA
#                    else:
#                        policy.policy[s, a] = eps / self.env.nA
        return values, policy

    def act(self, state):
        return None


# In[] TD(0) Discrete Model Free Predictor
class SarsaController(object):
    def __init__(self, env, alpha=0.01):
        if not issubclass(type(env), gym.envs.toy_text.discrete.DiscreteEnv):
            raise Exception('env should be subclass of gym.envs.toy_text.'
                            'discrete.DiscreteEnv')
        self.env = env
        self.alpha = alpha
        self.qvalues = np.zeros((self.env.nS, self.env.nA))
        self.qvalues = np.random.rand(self.env.nS, self.env.nA)/100

    def _action_from_q(self, qvalues, state, eps):
        rand = np.random.rand()
        if rand > eps:
            avalues = qvalues[state, :].ravel()
            a_star = np.argwhere(avalues == np.amax(avalues))
            a_star = a_star[0,0] # np.random.choice(a_star.ravel(), 1)[0]
            return a_star
        else:
            return np.random.choice(range(self.env.nA))

    def learn(self, iterations=1000, discount=1., eps=0.25):
        # if not isinstance(policy, StochasticDiscretePolicy):
        #     raise Exception('policy should have type DiscretePolicy')
        policy = StochasticDiscretePolicy(self.env)

        max_time_steps = 500

        # For each episode
        for i_episode in xrange(iterations):
            # Init state
            s = env.reset()
            # Select action for state using e-greedy
            a = self._action_from_q(self.qvalues, s, eps)  # policy.action(s)

            for t in xrange(1, max_time_steps):
                # observe next step and get reward
                sp, r, done, info = env.step(a)

                # sample a_prime random action
                ap = self._action_from_q(self.qvalues, sp, eps)

                td_target = r + discount * self.qvalues[sp, ap]
                self.qvalues[s, a] = self.qvalues[s, a] +\
                                     self.alpha * (td_target - self.qvalues[s, a])
                s = sp
                a = ap

                if done:
                    break

            if i_episode % 1000 == 0:
                msg = 'Episode {} finished in {} steps.'
                print msg.format(i_episode, t)

                policy.e_greedy_update(self.qvalues, eps)
                print(np.amax(self.qvalues, axis=1).reshape((8, 8)))
                p_pp(self.qvalues)

        policy.e_greedy_update(self.qvalues, eps)
        return self.qvalues, policy

# In[]
env_name = 'FrozenLake8x8-v0'  # 'Taxi-v1'
env = gym.make(env_name)

# In[]
np.random.seed(2)

mfp = SarsaController(env)
values, policy = mfp.learn(iterations=100000, discount=0.9, eps=0.5)

policy.policy
# In[]
print(values)
print(np.amax(values, axis=1).reshape((8, 8)))
#print(np.amax(policy.policy, axis=1).reshape((8, 8)))
p_pp(values)

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


# In[]
env_name = 'FrozenLake8x8-v0'  # 'Taxi-v1'
env = gym.make(env_name)

# In[]
np.random.seed(43)
policy = DiscretePolicy(env)

#policy.policy = ia.policy

#mfp = MonteCarloDMFController(env)
mfp = TD0DMFPredictor(env, alpha=0.01)
#mfp = TDlDMFPredictor(env, alpha=0.001, llambda=0.9)

values = mfp.evaluate(policy, iterations=10000, discount=1.0)

print values.reshape((8, 8))
print
#print ia.values.reshape((8, 8))

