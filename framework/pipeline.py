from .__init__ import *
import threading


class Pipeline:
    def __init__(self, modules: list[list[Module]]):
        self.modules = modules
        self.queue_list = []
        self.thread_pool = []
        self.lock_list: list[threading.Lock] = []
        self.build()

    def build(self):
        for stages in range(len(self.modules)):
            self.queue_list.append(TimePriorityQueue())
            self.lock_list.append(threading.Lock())
            for m in range(len(self.modules[stages])):
                self.modules[stages][m].set_output_queue(
                    self.queue_list[stages])
                self.modules[stages][m].set_output_lock(self.lock_list[stages])
                if stages != 0:
                    self.modules[stages][m].set_input_queue(
                        self.queue_list[stages-1])
                    self.modules[stages][m].set_input_lock(
                        self.lock_list[stages-1])
                self.thread_pool.append(threading.Thread(
                    target=self.modules[stages][m].run))

    def run(self):
        for i in self.thread_pool:
            i.start()
