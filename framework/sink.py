from .module import *


class Sink(Module):
    def __init__(self, name, max_queue_length=None):
        super().__init__(name, max_queue_length)
        self.time_slice = 0

    def process(self, packages: list[Package]):
        return NotImplementedError

    def run(self):
        while True:
            self.input_lock.acquire()
            if self.input_queue.is_empty() or self.input_queue.delta_time() < self.time_slice + 1:
                self.input_lock.release()
                continue
            packages = self.input_queue.get_time_slice(self.time_slice)
            self.input_lock.release()
            if len(packages) == 0:
                continue
            self.process(packages)
