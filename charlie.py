import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')

num_states = 1000
num_actions = env.action_space.n

# configuration
input_nodes = 10
layer1_nodes = 20
layer2_nodes = 40
output_nodes = 1
action_sets = [[1, 0], [0, 1]]
discount_rate = 0.9
learning_rate = 0.5


# inputs
prev_state = tf.placeholder(tf.float32, shape=[None, 4])
state = tf.placeholder(tf.float32, shape=[None, 4])
action = tf.placeholder(tf.float32, shape=[None, 2])

"""
    beginning of grpah
"""
inputs = tf.concat([state, prev_state-state, action], 1)
Q = tf.Variable(tf.zeros([input_nodes, layer1_nodes]), dtype=tf.float32)

# layers
layer1_weights = tf.Variable(tf.random_normal([input_nodes, layer1_nodes], mean=0.1), dtype=tf.float32)
layer1_bias = tf.Variable(tf.random_normal([layer1_nodes], mean=0.1), dtype=tf.float32)

layer2_weights = tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes], mean=0.1), dtype=tf.float32)
layer2_bias = tf.Variable(tf.random_normal([layer2_nodes], mean=0.1), dtype=tf.float32)

# output layer
output_weights = tf.Variable(tf.random_normal([layer2_nodes, output_nodes], mean=0.1), dtype=tf.float32)
output_bias = tf.Variable(tf.random_normal([output_nodes], mean=0.1), dtype=tf.float32)

# flow
layer1_outputs = tf.nn.dropout(tf.matmul(inputs, layer1_weights) + layer1_bias, 0.1)
layer2_outputs = tf.nn.dropout(tf.matmul(tf.nn.relu(layer1_outputs), layer2_weights) + layer2_bias, 0.1)
Q = tf.matmul(layer2_outputs, output_weights) + output_bias

"""
    end of graph
    beginning of logic
"""

Q_ = tf.placeholder(tf.float32, shape=[1, 1])
sqrt = tf.squared_difference(Q, Q_)
cross_entropy = tf.reduce_sum(sqrt)

trainer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


def run():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for episode in range(20000):
            pre_prev_observation = prev_observation = observation = env.reset()
            reward = 0
            current_action = [1, 0]
            max_Q = 0
            total_steps = total_steps if episode % 100 else 0

            for current_step in range(1000):
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
