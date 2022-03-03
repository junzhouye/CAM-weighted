import os
import csv


def mkdir_(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Log:
    def __init__(self, logRoot, fileName):
        mkdir_(logRoot)
        self.log_path = os.path.join(logRoot,fileName)

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path, 'a') as f:
            f.write(msg+"\n")


class LogCSV:
    def __init__(self, logRoot, fileName):
        mkdir_(logRoot)
        self.log_path = os.path.join(logRoot,fileName)

    def __call__(self, msg: list):
        print(msg, end='\n')
        with open(self.log_path, 'a',newline="") as f:
            writer = csv.writer(f)
            writer.writerow(msg)


if __name__ == "__main__":
    pass
    a = LogCSV("../log","a.csv")
    a(["acc","error","aws"])
    a([66,88,99])
    a([5,4,5])
    a([777,999,111])