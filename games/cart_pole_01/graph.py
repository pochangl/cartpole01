import tensorflow as tf
from reinforcement.graph import ReinforcementGraph


class CartPoleMixin:
    num_actions = 2
    action_space = list(range(num_actions))

    def define_inputs(self):
        self._state = tf.placeholder(tf.float32, shape=[None, 4], name='actual_states') # result state
        self.state = tf.placeholder(tf.float32, shape=[None, 4], name='state')

        self._reward = tf.placeholder(tf.float32, shape=[None], name='actual_reward')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')


class V1Graph(CartPoleMixin, Graph):
    def build_preprocess_graph(self):
        one_hot_action = tf.one_hot(self.action_space, self.action, name='one_hot_action')
        return tf.concat([one_hot_action, self.state], 1, name="bundled_input")

    def build_reward_graph(self):
        fcnn = self.build_fc_nn(input, [10, 20], tf.nn.relu)
        return tf.reduce_mean(fcnn, axis=1)
        
    def build_trainers(self):
        sqrt = tf.squared_difference(self._reward, self.reward, name='sqrt')
        cross_entropy = tf.reduce_sum(sqrt, name='cross_entropy')
        return tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='trainer'),

    def get_step_arguments(self, state):
        return {
            'fetches': [self.reward],
            'feed_dict': {
                self.state, state
            }
        }
    
    def get_training_arguments(self, training_batch):
        states = tuple(( batch.state for batch in training_batch ))
        _states = tuple(( batch._state for batch in training_batch ))
        actions = tuple(( batch.action for batch in training_batch ))
        _rewards = tuple(( batch._reward for batch in training_batch ))
        return {
            'fetches': self.trainers,
            'fetch_dict': {
                self.state: states,
                self._reward: _rewards,
                self._state: _states,
                self.action: actions,

            }
        }
