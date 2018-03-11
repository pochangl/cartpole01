import random
from collections import namedtuple

step = namedtuple(['state', '_state', 'action', '_reward'])

class ReinforcementHistory:
    history = []
    episode = []
    decay = 0.9
    batch_size = 50

    def __enter__(self):
        self.start()

    def __exit__(self, *args, **kwargs):
        self.end()

    def start(self):
        self.history = []
        self.episode = []

    def end(self):
        pass

    def get_mini_batch(self, batch_size):
        return random.sample(self.history, self.batch_size)

    def record(self, state, _state, action, reward):
        self.episode.append(step(
            state = state,
            _state = state,
            action = action,
            _reward = reward,
        ))

    def done(self, reward):
        length = len(self.episode)

        episode = [
            step(
                state = history.state,
                _state = history._state,
                action = history.action,
                _reward = (self.decay ** (length - index)) * reward
            )
            for index, history in enumerate(self.episode, 1)
        ]
        self.history.extend(episode)
        self.episode = []

