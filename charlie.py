import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
failures = []
num_actions = env.action_space.n

# configuration

action_sets = [[1, 0], [0, 1]]
learning_rate = 0.1

# inputs
_state = tf.placeholder(tf.float32, shape=[None, 4])
state = tf.placeholder(tf.float32, shape=[None, 4])
action = tf.placeholder(tf.float32, shape=[None, 2])

"""
    beginning of grpah
"""

def fdnn(inputs, sizes):
    # layers
    input_size = sizes[0]
    for size in sizes[1:]:
        weights = tf.Variable(tf.random_normal([input_size, size], mean=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.random_normal([size], mean=0.1), dtype=tf.float32)
        inputs = tf.nn.dropout(tf.matmul(inputs, weights) + bias, 0.1)
    return inputs

prediction_inputs = tf.concat([state, action], 1)
prediction = fdnn(prediction_inputs, [6, 10, 10, 4])
value = fdnn(prediction, [4, 1])
"""
    end of graph
    beginning of logic
"""

Q_ = tf.placeholder(tf.float32, shape=[1, 1])
sqrt = tf.squared_difference(state, prediction)
cross_entropy = tf.reduce_sum(sqrt)
prediction_trainer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

def train_failures():
    if not len(failures):
        return


def train_predictions():
    if not len(history)
        return


def run():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # first attempt
        for episode in range(20000):
            observation = env.reset()
            max_Q = 0
            total_steps = total_steps if episode % 100 else 0

            for current_step in range(1000):
                train_failures()
                train_predictions()
                prev_max_Q = max_Q
                prev_action = current_action
                pre_prev_observation, prev_observation = prev_observation, observation

                run_dict = {
                    state: np.tile(observation, (2, 1)),
                    prev_state: np.tile(prev_observation, (2, 1)),
                    action: [[1, 0], [0, 1]],
                }
                results, net = session.run([Q, layer2_outputs], feed_dict=run_dict)
                results = results.reshape([-1])
                max_Q = max(results)
                action_index = np.argmax(results)
                # print (action_index)
                # if episode % 100 == 99 and current_step == 0:
                #     print (results, max_Q, action_index)
                #     print (net)
                #     input()

                current_action = action_sets[action_index]
                observation, current_reward, done, info = env.step(action_index)
                observation = np.array(observation)
                if not done:
                    new_Q = prev_max_Q + learning_rate*(current_reward + discount_rate*(current_reward * max_Q) - prev_max_Q)
                else:
                    new_Q = -1
                train_dict = {
                    state: np.tile(prev_observation, (1, 1)),
                    prev_state: np.tile(pre_prev_observation, (1, 1)) - np.tile(prev_observation, (1, 1)),
                    action: [prev_action],
                    Q_: [[new_Q]],
                }
                if current_step >= 2:
                    _, loss = session.run((trainer, cross_entropy), feed_dict=train_dict)
                # print ('loss: %s' % loss)
                prev_action = current_action
                if done:
                    total_steps = total_steps + current_step
                    if episode % 100 == 99:
                        print ('\n\tepisode: %d average step: %f, loss: %f' % (episode, total_steps / 100, loss))
                    break
                prev_observation = observation
run()
