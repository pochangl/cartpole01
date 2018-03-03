import tensorflow as tf
import numpy as np
import gym

class CartPole01:
    env = gym.make('CartPole-v0')
    _states = [] # expected state
    states = [] # starting state
    actions = [] # actions
    edges = []
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))

    # configuration
    learning_rate = 0.1

    # inputs
    _state = tf.placeholder(tf.float32, shape=[None, 4], name='actual_states') # result state
    state = tf.placeholder(tf.float32, shape=[None, 4], name='state')
    action = tf.placeholder(tf.uint8, shape=[None], name='action')
    edge = tf.placeholder(tf.float32, shape=[None, 4], name='edges')

    """
        beginning of grpah
    """

    def fdnn(self, inputs, sizes):
        # layers
        input_size = sizes[0]
        for size in sizes[1:]:
            weights = tf.Variable(tf.random_normal([input_size, size], mean=0), dtype=tf.float32)
            bias = tf.Variable(tf.random_normal([size], mean=0), dtype=tf.float32)
            inputs = tf.matmul(inputs, weights) + bias
            input_size = size
        return inputs

    def build(self):
        self.env = env = gym.make('CartPole-v0')
        one_hot_action = tf.one_hot(self.action, depth=self.num_actions)

        prediction_inputs = tf.concat([self.state, one_hot_action], 1)
        self.prediction = prediction = self.fdnn(prediction_inputs, [6, 4, 4])
        normalized_edge = tf.nn.l2_normalize(self.edge, axis=0)
        print (normalized_edge)

        self.value = tf.reduce_sum(tf.matmul(prediction, tf.transpose(normalized_edge)), axis=1)

        self.decision = tf.argmin(self.value)

        sqrt = tf.squared_difference(self._state, prediction)
        self.prediction_cross_entropy = cross_entropy = tf.reduce_sum(sqrt)
        self.prediction_trainer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    def train(self):
        train_dict = {
            self.state: self.states,
            self._state: self._states,
            self.action: self.actions,
        }
        for _ in range(100):
            _, prediction_loss = self.session.run([self.prediction_trainer, self.prediction_cross_entropy], feed_dict=train_dict)
        length = len(self.states)
        print ('loss: %10d length: %5d' % (prediction_loss/length, length), end='\t')
        self.states = []
        self._states = []
        self.values = []
        self.actions = []

    def step(self, observation, step):
        if not self.edges:
            decision = self.env.action_space.sample()
            observation = tuple(self.env.step(decision))
        else:
            states = np.tile(observation, (self.num_actions, 1))
            run_dict = {
                self.state: states,
                self.action: self.all_actions,
                self.edge: self.edges,
            }
            decision, value = self.session.run([self.decision, self.value], feed_dict=run_dict)
            # print (value)
            observation = tuple(self.env.step(decision))
        return decision, observation


    def start(self):
        with tf.Session() as session:
            self.session = session
            self.build()
            session.run(tf.global_variables_initializer())
            # first attempt
            for episode in range(50000):
                observation = self.env.reset()
                if not episode % 100:
                    total_steps = 0

                for step in range(1000):
                    action, (new_observation, reward, done, info) = self.step(observation, step)
                    self._states.append(new_observation)
                    self.states.append(observation)
                    self.actions.append(action)
                    observation = new_observation

                    if done:
                        # print (observation)
                        self.edges.append(observation)
                        total_steps = total_steps + step
                        if episode % 100 == 99:
                            ones = list(filter(lambda v: v, self.actions))
                            print ('action ratio: %1.2f' % (len(ones)*1.0/ len(self.actions)), end=' ')
                            self.train()
                            print ('episode: %10d average step: %10f' % (episode, total_steps / 100))
                        break
cp = CartPole01()
cp.start()
