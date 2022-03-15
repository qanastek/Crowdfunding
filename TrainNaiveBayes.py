from Trainer import Trainer

class TrainNaiveBayes(Trainer):

    def __init__(self, path):
        super().__init__(path)

    def train(self):
        print("Please implement the train function first in TrainNaiveBayes!")
        self.model = None
        raise NotImplementedError

s = TrainNaiveBayes("data/projects.csv")
s.train()