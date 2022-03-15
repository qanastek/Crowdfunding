from Trainer import Trainer

class TrainKNN(Trainer):

    def __init__(self, path, save_gzip_path=None):
        super().__init__(path, save_gzip_path=save_gzip_path)

    def train(self):
        print("Please implement the train function first in TrainKNN!")
        self.model = None
        raise NotImplementedError

s = TrainKNN("data/projects.csv", save_gzip_path="data/knn-prepro")
s.train()