import tensorflow as tf
import numpy as np
import gym
import random


class CartPole01:
    env = gym.make('CartPole-v0')
    states = [] # starting state
    _states = [] # expected state
    actions = [] # actions
    dones = []
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))
    prediction_loss = 10000
    done_loss = 10000
    value_loss = 10000

    def reset(self):
        self.states = []
        self._states = []
        self.actions = []
        self.values = []
        self.dones = []

    # configuration
    learning_rate = 0.1

    # inputs
    _state = tf.placeholder(tf.float32, shape=[None, 4], name='actual_states') # result state
    state = tf.placeholder(tf.float32, shape=[None, 4], name='state')
    action = tf.placeholder(tf.int32, shape=[None], name='action')
    _value = tf.placeholder(tf.float32, shape=[None], name='actual_value')
    _done = tf.placeholder(tf.float32, shape=[None], name='actual_value')

    """
        beginning of grpah
    """

    def fcnn(self, inputs, sizes):
        # layers
        input_size = sizes[0]
        for size in sizes[1:]:
            weights = tf.Variable(tf.random_normal([input_size, size], mean=0), dtype=tf.float32)
            bias = tf.Variable(tf.random_normal([size], mean=0), dtype=tf.float32)
            inputs = tf.nn.relu(tf.matmul(inputs, weights) + bias)
            input_size = size
        return inputs

    def build(self):
        print ('building network')
        self.env = gym.make('CartPole-v0')
        one_hot_action = tf.one_hot(self.action, depth=self.num_actions)

        bundled_inputs = tf.concat([self.state, one_hot_action], 1, name='bundled_input')
        print(bundled_inputs)
        self.predicted_state = self.fcnn(bundled_inputs, [6, 16, 4])

        sqrt = tf.squared_difference(self._state, self.predicted_state)
        self.predicted_state_cross_entropy = tf.reduce_sum(sqrt)
        self.predicted_state_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.predicted_state_cross_entropy)

        self.dones_ = dones = tf.sigmoid(self.fcnn(bundled_inputs, [6, 12, 24, 1]))
        print (dones)

        self.done = tf.reduce_max(dones, axis=1)
        print (self.done)
        sqrt = tf.squared_difference(self._done, self.done)
        self.done_cross_entropy = tf.reduce_sum(sqrt)
        self.done_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.done_cross_entropy)
        self.dones_count = tf.reduce_sum(self.done)
        print(self.dones_count)

        self.value = self.fcnn(bundled_inputs, [6, 12, 24, 1])
        print (self.value)
        sqrt = tf.squared_difference(self._value, self.value)
        self.value_cross_entropy = tf.reduce_sum(sqrt)
        self.value_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.value_cross_entropy)

        self.decision = tf.argmax(self.value)
        print ('network built')

    def search(self, states, action, iterations = 40):
        if iterations <= 0:
            return 0

        for i in range(iterations):
            feed_dict = {
                self.state: states,
                self.action: [action],
            }
            _states, [done] = self.session.run([self.predicted_state, self.done], feed_dict=feed_dict)
            if round(done):
                break
            states = _states
        return i + 1

    def train(self):
        values = [self.search(states=[state], action=action) for state, action in zip(self.states, self.actions)]
        train_dict = {
            self.state: self.states,
            self._state: self._states,
            self.action: self.actions,
            self._done: self.dones,
            self._value: values,
        }
        assert len(self.states) == len(self._states) == len(self.actions) == len(self.dones) == len(values)
        for _ in range(500):
            _, prediction_loss, _, done_loss, _, value_loss = self.session.run([self.predicted_state_trainer, self.predicted_state_cross_entropy, self.done_trainer, self.done_cross_entropy, self.value_trainer, self.value_cross_entropy], feed_dict=train_dict)

        print()
        print(self.dones[:10])
        print(values[:10])
        print ('state prediction loss: %5d\t done loss: %5d\t value loss: %5d' % (prediction_loss, done_loss, value_loss), end='\t')
        self.reset()

    def make_one_step(self, observation, step):
        states = np.tile(observation, (self.num_actions, 1))
        run_dict = {
            self.state: states,
            self.action: self.all_actions,
        }
        decision, [decision], value = self.session.run([self.decision, self.decision, self.value], feed_dict=run_dict)
        if (self.episode % 1000 and step == 0):
            print ('decision: ', value, decision)
        # print (value)
        if self.episode % 5:
           decision = 1 - decision
        observation = tuple(self.env.step(decision))
        return decision, observation

    def start(self):
        self.episode = 0
        with tf.Session() as session:
            self.session = session
            self.build()
            session.run(tf.global_variables_initializer())
            # first attempt
            for episode in range(50000):
                observation = self.env.reset()
                if not episode % 100:
                    total_steps = 0
                self.episode = episode

                for step in range(1000):
                    action, (new_observation, reward, done, info) = self.make_one_step(observation, [step])
                    if step > 100:
                        self.env.render()

                    if done:
                        self._states.append(observation)
                        self.states.append(last_observation)
                        self.actions.append(last_action)
                        self.dones.append(0)
                        self._states.append(new_observation)
                        self.states.append(observation)
                        self.actions.append(action)
                        self.dones.append(1)
                        # print (observation)
                        total_steps = total_steps + step
                        if episode % 500 == 499:
                            ones = list(filter(lambda v: v, self.actions))
                            print ('action ratio: %1.2f' % (len(ones)*1.0/ len(self.actions)), end=' ')
                            self.train()
                            print ('episode: %10d average step: %10f' % (episode, total_steps / 100))
                        break

                    last_observation = observation
                    last_action = action
                    observation = new_observation
cp = CartPole01()
cp.start()
