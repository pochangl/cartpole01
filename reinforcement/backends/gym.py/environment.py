from reinforcement.environment import BaseEnvironment
import gym


class GameNotSpecifiedError(NotImplementedError):
  pass


class OpenAIGymEnvironment(BaseEnvironment):
  '''
    open ai gym environment
  '''
  name = None

  def __init__(self, name=None):
    super().__init__()
    if name is not None:
      self.env = self.name

  def get_environment(self):
    if self.env:
      return self.env
    elif self.name:
      return gym.make(self.name)
    else:
      raise GameNotSpecifiedError('OpenAIGymStrategy requires either name or env')

  def reset(self):
    self._reset()
    self.initial_states = self.state = self.env.reset()

  def observe(self):
    return self.state

  def step(self, *args, **kwargs):
    return self.env.step(*args, **kwargs)