'''
    Draph is TensorFlow graph management Class
'''
import tensorflow as tf


class ReinforcementGraph:
    '''
        graph class for manipulating TensorFlow graph
    '''
    iput = None
    pre_iput = None
    oput = None
    build_decision = None
    trainers = None
    ipt = None

    def __init__(self):
        self.build()

    def define_inputs(self):
        '''
            define input place holders here
        '''
        raise NotImplementedError()

    def build_preprocess_graph(self):
        '''
            process input for better manipulation
        '''
        return self.ipt

    def build_reward_graph(self):
        '''
            build main logic
        '''

    def build_trainers(self):
        '''
            get trains
        '''
        raise NotImplementedError()

    def build(self):
        '''
            build the overall graph
        '''
        self.define_inputs()
        self.preprocessed_input = self.build_preprocess_graph()
        self.reward = self.build_reward_graph()
        self.trainers = self.build_trainers()
        assert issubclass(self.trainers, [list, tuple])

    def build_fc_nn(self, input, sizes, activation_function, prefix=''):
        # layers
        input_size = int(input.shape[-1])
        for layer, size in enumerate(sizes, 1):
            weights = tf.Variable(tf.random_normal([input_size, size], mean=0), dtype=tf.float32, name='layer_%d_weight' % layer)
            bias = tf.Variable(tf.random_normal([size], mean=0), dtype=tf.float32, name='layer_%d_bias' % layer)
            input = activation_function(tf.matmul(input, weights) + bias, name='layer_%d_output' % layer)
            input_size = size
        return input
    
    def get_step_arguments(self, state):
        raise NotImplementedError()

    def get_training_arguments(self, observations):
        raise NotImplementedError()