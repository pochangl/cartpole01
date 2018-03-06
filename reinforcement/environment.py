
class BaseEnvironment:
  env = None

  def __enter__(self):
    self.env = self.get_environment()
    return self

  def __exit__(self, type, value, traceback):
    self.close()

  def get_environment(self):
    raise NotImplementedError

  def destroy_environment(self):
    raise NotImplementedError

  def step(self):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError

  def observe(self):
    raise NotImplementedError

  def close(self):
    self.destroy_environment()
