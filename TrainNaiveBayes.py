from Trainer import Trainer
from sklearn.naive_bayes import GaussianNB

class TrainNaiveBayes(Trainer):

    def __init__(self, path, save_gzip_path=None, clean_gzip=False):
        super().__init__(path)
        self.model = GaussianNB()

    def train(self):

        self.model.fit(self.ds.x_train, self.ds.y_train)

        res = self.predict()
        print(res)

        print("Score train: ", self.model.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", self.model.score(self.ds.x_test, self.ds.y_test))

s = TrainNaiveBayes("data/short.csv", save_gzip_path="data/bayes-prepro")
# s = TrainNaiveBayes("data/projects.csv", save_gzip_path="data/bayes-prepro")
s.train()