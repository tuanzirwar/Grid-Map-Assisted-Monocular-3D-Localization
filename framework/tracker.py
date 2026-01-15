from .module import *
import time


class Tracker(Module):
    def __init__(self, name, max_queue_length=None):
        super().__init__(name, max_queue_length)
        self.max_queue_length = max_queue_length
        self.time_slice = 0
        if max_queue_length != None:
            self.output_queue.set_max_count(max_queue_length)

    def process(self, packages: list[Package]) -> list[Package]:
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
            packages = self.process(packages)

            for package in packages:
                while self.output_queue.is_full():
                    time.sleep(0.1)
                self.output_lock.acquire()
                self.output_queue.push(package)
                self.output_lock.release()
