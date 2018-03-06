'''
    agent module
        agent are responsible for playing games
        like act, train, and observe
'''

import collections
import random

Observations = collections.namedtuple('Observations', ['state', 'reward', 'done', 'info', 'step'])


class EnvironmentClassNotSetError(NotImplementedError):
    '''
        environment class not set error
    '''


class GraphClassNotSetError(NotImplementedError):
    '''
        graph class not set error
    '''


class Agent:
    environment_class = None
    graph_class = None
    env = None
    batch_size = 50
    graph = None
    history = []
    episode = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.end()

    def get_environment(self):
        '''
            env getter
        '''
        if self.environment_class is None:
            raise EnvironmentClassNotSetError('environment_class is not set')
        return self.environment_class() # pylint: disable=E1102

    def get_graph(self):
        '''
            graph getter
        '''
        if self.graph_class is None:
            raise GraphClassNotSetError('graph_class is not set')
        return self.graph_class() # pylint: disable=E1102

    def start(self):
        '''
            create a strategy, but throw error if already did
        '''
        if self.env is not None:
            self.env = self.get_environment()
        return self

    def end(self):
        '''
            end strateg and close environment
        '''
        self.env.close()
        self.env = None

    def get_feed_dict(self):
        '''
            dict for running the step
        '''
        return {
            self.graph.get_iput(): self.env.observe()
        }

    def get_minibatch(self):
        return random.sample(self.history, self.batch_size)

    def get_training_feed_dict(self):
        observations = self.get_minibatch()
        iputs = (obs.state for obs in observations)
        _oputs = (obs.state for obs in observations)

        return {
            self.graph.get_iput(): iputs,
            self.graph.get__oput(): _oputs,
        }

    def train(self, session):
        '''
            train the model
        '''
        return session.run(self.graph.get_trainers(), feed_dict=self.get_training_feed_dict())

    def step(self):
        raise NotImplementedError

    def run(self):


class QLearningStrategy(Agent):
    decay = 0.9
