from Trainer import Trainer

class TrainSVM(Trainer):

    def __init__(self, path):
        super().__init__(path)

    def train(self):
        print("Please implement the train function first in TrainSVM!")
        self.model = None
        raise NotImplementedError

s = TrainSVM("data/projects.csv")
s.train()