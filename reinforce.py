# In[]
import gym
import numpy as np
import theano
import theano.tensor as T
import lasagne
import sklearn.preprocessing
np.set_printoptions(precision=2)


from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion

# In[]
class ValueFunctionApproximator:
    def __init__(self, env, batch_size, learning_rate):
        self.nA = env.action_space.n
        self.sS = env.observation_space.shape[0]
        self.batch_size = batch_size
        self.lr = theano.shared(np.float32(learning_rate))
        self._init_model()
        self.env = env

        observation_examples = np.array([env.observation_space.sample() for x in range(100000)])

        # Fit feature scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

    def _init_model(self):
        self.nn_x, self.nn_z = T.matrices('x', 'z')
        self.nn_lh1 = lasagne.layers.InputLayer(shape=(None, self.sS),
                                             input_var=self.nn_x)
        self.nn_lh2 = lasagne.layers.DenseLayer(self.nn_lh1, 32,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                                W=lasagne.init.GlorotNormal(),
                                                b=lasagne.init.Constant(0.))

        self.nn_lh3 = lasagne.layers.DenseLayer(self.nn_lh2, 8,
                                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                                W=lasagne.init.GlorotNormal(),
                                                b=lasagne.init.Constant(0.))

        self.nn_ly = lasagne.layers.DenseLayer(self.nn_lh3, self.nA,
                                               nonlinearity=lasagne.nonlinearities.linear,
                                               W=lasagne.init.Normal(),
                                               b=lasagne.init.Constant(0.))
        self.nn_y = lasagne.layers.get_output(self.nn_ly)

        self.f_predict = theano.function([self.nn_x], self.nn_y)

        self.nn_params = lasagne.layers.get_all_params(self.nn_ly, unwrap_shared=False, trainable=True)

        self.nn_cost = T.sum(lasagne.objectives.squared_error(self.nn_y, self.nn_z))

        #self.nn_updates = lasagne.updates.sgd(self.nn_cost, self.nn_params, learning_rate=self.lr)
        self.nn_updates = lasagne.updates.rmsprop(self.nn_cost, self.nn_params, learning_rate=self.lr)
        #self.nn_updates = lasagne.updates.adam(self.nn_cost, self.nn_params)
        self.f_train = theano.function([self.nn_x, self.nn_z],
                                       [self.nn_y, self.nn_cost],
                                       updates=self.nn_updates)

    def _scale_state(self, s_float32):
        return self.scaler.transform(s_float32)

    def predict(self, s):
        s_float32 = np.array(s)
        if len(s_float32.shape) == 1:
            s_float32 = np.expand_dims(s_float32, axis=0)
        if len(s_float32.shape) != 2:
            raise RuntimeError('Input should be an 2d-array or row-vector.')
        #s_float32 = self._scale_state(s_float32)
        #s_float32 = self.feature_map.transform(s_float32)
        s_float32 = s_float32.astype(np.float32)
        return self.f_predict(s_float32)

    def train(self, states, actions, rewards):
        s_float32 = np.array(states).astype(np.float32)
        if len(s_float32.shape) == 1:
            s_float32 = np.expand_dims(s_float32, axis=0)
        if len(s_float32.shape) != 2:
            raise RuntimeError('Input should be an 2d-array or row-vector.')
        #s_float32 = self._scale_state(s_float32)
        #s_float32 = self.feature_map.transform(s_float32)
        s_float32 = s_float32.astype(np.float32)
        a_float32 = np.array(actions).astype(np.float32)
        result = self.f_train(s_float32, a_float32)
        return result


# In[]
class Agent:

    def __init__(self, env, eps=1.0, learning_rate=0.1):
        self.nA = env.action_space.n
        self.eps = eps
        self.value_function = ValueFunctionApproximator(env, 32, learning_rate)

    def q_values(self, s):
        return self.value_function.predict(s)

    def act(self, s):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.nA)
        else:
            return np.argmax(self.value_function.predict(s))

    def estimate(self, s, a):
        prediction = self.value_function.predict(s)[0]
        return prediction[a]

    def learn(self, s, targets):
        self.value_function.train(s, targets, [])


# In[]
class ReplayMemory:

    def __init__(self, agent, capacity):
        self.agent = agent
        self.capacity = capacity
        self.memory = []

    # State, action, reward and next state
    def append(self, s, a, r, sp, done):
        self.memory.append([s, a, r, sp, int(done)])

        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size, discount=1.0):
        batch_size = min(batch_size, len(self.memory))
        choices = np.random.choice(len(self.memory), batch_size)
        s = np.array([self.memory[i][0] for i in choices])
        a = np.array([self.memory[i][1] for i in choices])
        r = np.array([self.memory[i][2] for i in choices])
        sp = np.array([self.memory[i][3] for i in choices])
        done = np.array([self.memory[i][4] for i in choices])

        q_vals = agent.q_values(s)
        target = r + (1 - done) * discount * np.amax(agent.q_values(sp), axis=1)
        for i in range(len(choices)):
            q_vals[i, a[i]] = target[i]

        return s, q_vals

# In[] Plotting stuff
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

plt.ion()
fig = plt.figure()
def plot_qsurface(agent):
    N = 15
    ticks = [0 for i in range(len(env.low))]
    for ax in range(len(env.low)):
        ticks[ax] = np.linspace(env.low[ax], env.high[ax], N)

    z = np.zeros((N, N))
    for i, x in enumerate(ticks[ax]):
        for j, y in enumerate(ticks[ax]):
            #print [x, y], agent.q_values([x, y])
            z[i, j] = -np.max(agent.q_values([x, y])[0])
    #        z[i, j] = -agent.q_values([x, y])[0, 2]

    ax = fig.gca(projection='3d')
    ax.cla()
    X, Y = np.meshgrid(ticks[0], ticks[1])
    surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1,
                           linewidth=0, antialiased=True)
    plt.draw()
    plt.show()
    plt.pause(0.001)

# In[]
env_name = 'CartPole-v1' #'MountainCar-v0'
env = gym.make(env_name)

# In[]
done = False
agent = Agent(env, eps=0.5, learning_rate=0.0001)
memory = ReplayMemory(agent, 100000)
discount = 0.99 # 1.0
#plot_qsurface(agent)


# In[]
if False:
    agent.eps = 0.2 #2
    agent.value_function.lr.get_value()
    agent.value_function.lr.set_value(0.0001)

# In[] Main Q Learning loop
render = False
n_episodes = 10000
max_steps_per_episode = 1000
for episode in range(n_episodes):
    steps = 0
    s = env.reset()
    done = False
    while not done:
        if render: env.render()
        a = agent.act(s)
        q_vals = agent.q_values(s)

        sp, r, done, info = env.step(a)
        memory.append(s, a, r, sp, done)

        if len(memory.memory) > 128:
            mem_states, mem_targets = memory.sample(64, discount)
            mem_states = np.array(mem_states)
            mem_targets = np.array(mem_targets)
            agent.learn(mem_states, mem_targets)

        if steps % 500 == 0:
            print('Episode {}, Step {}, eps {}'.format(episode, steps, agent.eps))
            if len(memory.memory) > 128:
                print('\t{}'.format(mem_targets[0]))

        if done or steps > max_steps_per_episode:
            print("Episode finished after {} timesteps".format(steps))
            print()
            break
        s = sp
        steps += 1

        if agent.eps >= 0.01 and steps % 10000 == 0:
            agent.eps *= 0.9

    if agent.eps >= 0.0:
        agent.eps *= 0.999

# In[] Act

monitoring = True
render = False
monitor_name = './' + env_name + '-' + 'qlearning' + '-experiment'
if monitoring:
    env.monitor.start(monitor_name, force=True)

for e in range(150):
    s = env.reset()
    episode = 0
    done = False
    #tmp = agent.eps
    #agent.eps = 0.0
    while not done and episode < 500:
        if render: env.render()
        a = agent.act(s)
        sp, r, done, info = env.step(a)
        s = sp
        episode += 1
    #agent.eps = tmp
    print('episode {} finished in {} steps'.format(e, episode))

if monitoring:
    env.monitor.close()

# In[]
gym.upload(monitor_name, api_key='sk_3MIHM0n0QseEaRQH7mVtxQ', ignore_open_monitors=True)
