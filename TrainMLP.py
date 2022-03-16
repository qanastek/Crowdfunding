from Trainer import Trainer

# class MLP:



class TrainMLP(Trainer):

    def __init__(self, path, save_gzip_path=None, clean_gzip=False):
        super().__init__(path)
        self.model = None

    def train(self):

        self.model.fit(self.ds.x_train, self.ds.y_train)

        res = self.predict()
        print(res)

        print("Score train: ", self.model.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", self.model.score(self.ds.x_test, self.ds.y_test))

s = TrainMLP("data/short.csv", save_gzip_path="data/mlp-prepro")
# s = TrainMLP("data/projects.csv", save_gzip_path="data/mlp-prepro")
s.train()