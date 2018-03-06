from reinforcement.strategy import BaseStrategy


class OpenAIStrategy(BaseStrategy):
  history = []
  episode_states = []
  step_number = 0

  def _reset(self):
    self.episode_states = []
    self.step_number = 0
