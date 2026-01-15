from .module import *
import time


class Source(Module):
    def __init__(self, name, max_queue_length=None):
        super().__init__(name,  max_queue_length)

    def process(self):
        return NotImplementedError

    def run(self):
        while True:
            packages= self.process()
            while self.output_queue.is_full():
                time.sleep(0.1)
            for package in packages:
                while self.output_queue.is_full():
                    time.sleep(0.1)
                self.output_lock.acquire()
                self.output_queue.push(package)
                self.output_lock.release()
            
