import tensorflow as tf
import numpy as np
import gym

class CartPole01:
    env = gym.make('CartPole-v0')
    _states = [] # expected state
    states = [] # starting state
    actions = [] # actions
    values = []
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))

    # configuration
    learning_rate = 0.1

    # inputs
    _state = tf.placeholder(tf.float32, shape=[None, 4], name='actual_states') # result state
    state = tf.placeholder(tf.float32, shape=[None, 4], name='state')
    action = tf.placeholder(tf.uint8, shape=[None], name='action')
    _value = tf.placeholder(tf.float32, shape=[None], name='actual_value')

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
        self.prediction = prediction = self.fdnn(prediction_inputs, [6, 10, 4])
        value = self.fdnn(prediction, [4, 1])
        self.decision = tf.argmax(value)

        sqrt = tf.squared_difference(self._state, prediction)
        self.prediction_cross_entropy = cross_entropy = tf.reduce_sum(sqrt)
        self.prediction_trainer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        sqrt = tf.squared_difference(value, self._value)
        self.value_cross_entropy = cross_entropy = tf.reduce_sum(sqrt)
        self.value_trainer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    def train(self):
        train_dict = {
            self.state: self.states,
            self._state: self._states,
            self.action: self.actions,
            self._value: self.values,
        }
        for _ in range(100):
            _, _, value_loss, prediction_loss = self.session.run([self.value_trainer, self.prediction_trainer, self.value_cross_entropy, self.prediction_cross_entropy], feed_dict=train_dict)
        print ('prediction: %10d value: %10d length: %5d' % (prediction_loss/len(self.values),  value_loss/len(self.values), len(self.values)), end='\t')
        self.states = []
        self._states = []
        self.values = []
        self.actions = []

    def step(self, observation, step):
        run_dict = {
            self.state: np.tile(observation, (self.num_actions, 1)),
            self.action: self.all_actions
        }
        ((decision,), ) = self.session.run([self.decision], feed_dict=run_dict)
        observation = tuple(self.env.step(decision))
        return decision, observation


    def start(self):
        with tf.Session() as session:
            self.session = session
            self.build()
            session.run(tf.global_variables_initializer())
            # first attempt
            for episode in range(50000):
                values = []
                observation = self.env.reset()
                if not episode % 100:
                    total_steps = 0

                for step in range(1000):
                    action, (new_observation, reward, done, info) = self.step(observation, step)
                    self._states.append(new_observation)
                    self.states.append(observation)
                    values.append(step)
                    self.actions.append(action)
                    observation = new_observation

                    if done:
                        total_steps = total_steps + step
                        values = map(lambda v: v/step, values)
                        self.values.extend(values)
                        values = []
                        if episode % 100 == 99:
                            ones = list(filter(lambda v: v, self.actions))
                            print ('action ratio: %f' % (len(ones)*1.0/ len(self.actions)), end=' ')
                            self.train()
                            print ('episode: %10d average step: %10f' % (episode, total_steps / 100))
                        break
cp = CartPole01()
cp.start()
