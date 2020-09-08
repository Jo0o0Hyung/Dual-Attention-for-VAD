import os
from tensorboard_logger import configure, log_value


class Logger(object):
    def __init__(self, log_dir):
        self._remove(log_dir)

        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)

        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        '''param <path> could either be relative or absolute'''
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path) # remove dir and all contains
