from Trainer import Trainer

class TrainKNN(Trainer):

    def __init__(self, path):
        super().__init__(path)

    def train(self):
        print("Please implement the train function first in TrainKNN!")
        self.model = None
        raise NotImplementedError

s = TrainKNN("data/projects.csv")
s.train()