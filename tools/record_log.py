import os
import time


class Log:
    def __init__(self, log_path, log_name=None, eveytime=True, enable=True):
        self.enable = enable
        self.log_print = None  # function  args = (content, key ,end="\n")
        
        if not self.enable:
            return
        self.log_path = log_path
        time_recoder = time.strftime("%Y%m%d_%Hh%Mm%Ss", time.localtime())
        self.eveytime = eveytime

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if log_name is None:
            self.log_file = os.path.join(self.log_path, f"{time_recoder}.log")
        else:
            self.log_file = os.path.join(self.log_path, log_name)

        if not eveytime:
            self.f = open(self.log_file, "a+", encoding="utf-8")

    
    def record(self, content):
        if self.log_print is not None:
            self.log_print(content)
            
        if not self.enable:
            return
        if self.eveytime:
            with open(self.log_file, "a") as f:
                f.write(content+"\n")
        else:
            self.f.write(content+"\n")

    def log_show(self, log_print):
        self.log_print = log_print

    def close_record(self):
        try:
            self.f.close()
        except:
            pass
        print("Log file closed!")
