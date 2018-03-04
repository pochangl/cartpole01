import tensorflow as tf
import numpy as np
import gym


class CartPole01:
    env = gym.make('CartPole-v0')
    states = [] # starting state
    _states = [] # expected state
    steps = []
    _steps = []
    actions = [] # actions
    dones = []
    edges = []
    num_actions = env.action_space.n
    all_actions = list(range(num_actions))

    def reset(self):
        self.states = []
        self._states = []
        self._steps = []
        self.actions = []
        self.values = []
        self.steps = []
        self.dones = []

    # configuration
    learning_rate = 0.1

    # inputs
    step = tf.placeholder(tf.float32, shape=[None, 1], name='step')
    _step = step + 1
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
            inputs = tf.matmul(inputs, weights) + bias
            input_size = size
        return inputs

    def build(self):
        print ('building network')
        self.env = gym.make('CartPole-v0')
        one_hot_action = tf.one_hot(self.action, depth=self.num_actions)

        bundled_inputs = tf.concat([self.state, one_hot_action], 1, name='bundled_input')
        self.predicted_state = self.fcnn(bundled_inputs, [6, 4, 4])

        sqrt = tf.squared_difference(self._state, self.predicted_state)
        self.predicted_state_cross_entropy = tf.reduce_sum(sqrt)
        self.predicted_state_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.predicted_state_cross_entropy)

        stepped_bundled_input = tf.concat([self.state, one_hot_action, self.step], 1, name="stepped_input")
        self.dones = self.fcnn(stepped_bundled_input, [7, 1, 10])

        self.done = tf.reduce_max(tf.nn.relu(tf.sign(self.dones)))
        sqrt = tf.squared_difference(self._done, self.done)
        self.done_cross_entropy = tf.reduce_sum(sqrt)
        self.done_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.done_cross_entropy)
        self.dones_count = tf.count_nonzero(self.done)

        self.value = self.fcnn(stepped_bundled_input, [7, 17, 34, 1])
        sqrt = tf.squared_difference(self._value, self.value)
        self.value_cross_entropy = tf.reduce_sum(sqrt)
        self.value_trainer = tf.train.AdamOptimizer(1e-4).minimize(self.value_cross_entropy)

        self.decision = tf.argmax(self.value)
        print ('network builded')

    def search(self, states, step, iterations = 8, action_sets={}):
        if iterations <= 0:
            return 0
        length = len(states)
        states = states * 2
        try:
            actions = action_sets[length]
        except:
            actions = action_sets[length] = [0] * length + ([1] * length)
        feed_dict = {
            self.state: states,
            self.action: actions,
            self.step: [step] * (length * 2)
        }
        _states, count = self.session.run([self.predicted_state, self.dones_count], feed_dict=feed_dict)
        cnt = self.search(states = states, step = step + 1, iterations=iterations-1)
        assert isinstance(count, float)
        return count + cnt


    def train(self):
        values = [self.search(states=[state], step=step) for state, action, step in zip(self.states, self.actions, self.steps)]
        train_dict = {
            self.state: self.states,
            self._state: self._states,
            self.action: self.actions,
            self.step: self.steps,
            self.done: self.dones,
            self._value: values,
        }
        for _ in range(100):
            _, prediction_loss, _, done_loss, _, value_loss = self.session.run([self.predicted_state_trainer, self.predicted_state_cross_entropy, self.done_trainer, self.done_cross_entropy, self.value_trainer, self.value_cross_entropy], feed_dict=train_dict)
        length = len(self.states)
        print ('loss: %5d\t length: %5d\t done loss: %5d\t value loss: %5d' % (prediction_loss/length, length, done_loss, value_loss), end='\t')
        self.reset()

    def make_one_step(self, observation, step):
        states = np.tile(observation, (self.num_actions, 1))
        run_dict = {
            self.state: states,
            self.action: self.all_actions,
            self.step: [step],
        }
        decision, decision = self.session.run([self.decision, self.decision], feed_dict=run_dict)
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
                    action, (new_observation, reward, done, info) = self.make_one_step(observation, [step])
                    self._states.append(new_observation)
                    self.states.append(observation)
                    self.actions.append(action)
                    self.dones.append(done)
                    self.steps.append([step])
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
