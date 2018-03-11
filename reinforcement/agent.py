'''
    agent module
        agent are responsible for playing games
        like act, train, and observe
'''
import tensorflow as tf
import collections
import random

Observations = collections.namedtuple('Observations', ['state', '_reward', 'done'])


class Agent:
    environment_class = None
    graph_class = None
    history_class
    env = None
    batch_size = 50
    graph = None
    history = None
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

    def get_history(self):
        '''
            history getter
        '''
        if self.history_class is None:
            raise NotImplementedError('history_class is not set')
        return self.history_class() # pylint: disable=E1102

    def start(self):
        '''
            create a strategy, but throw error if already did
        '''
        if self.env is not None:
            self.env = self.get_environment()
        if self.history is not None:
            self.history = self.get_hisotry()
        if self.graph is not None:
            self.history = self.get_graph()
        return self

    def end(self):
        '''
            end strateg and close environment
        '''
        if self is not None:
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
        return self.history.get_minibatch(self.batch_size)

    def train(self, session):
        '''
            train the model
        '''
        return session.run(**self.graph.get_training_artuments(self.env.observe))

    def reset(self):
        self.env.reset()
        return self.env.observe()

    def make_decision(self, result):
        return result

    def step(self, session):
        state = self.env.observe()
        self.get_feed_dict()
        result = session.run(**self.graph.get_step_arguments(state))
        decision = self.make_decision(result)
        obs = self.env.step(decision)
        self.history.record(state=obs.state, _state=obs._state, action=decision, _reward=0)
        return obs

    def episode_end(self):
        self.history.done()
        self.env.reset()

    def is_done(self, observations):
        return observations.done

    def steps(self, session):
        done = False
        while not done:
            obs = self.step(session)
            yield obs
            done = obs.is_done

    def run(self, iterations):
        session = tf.Session()
        with self, session:
            session.run(tf.global_variables_initializer())
            for episode in range(iterations):
                self.episode_number  = episode
                self.step_number = 0
                self.reset()

                for observation in self.steps(session):
                    pass
                else:
                    self.episode_end()
