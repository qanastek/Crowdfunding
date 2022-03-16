from Trainer import Trainer
from sklearn.neighbors import KNeighborsClassifier

class TrainKNN(Trainer):

    def __init__(self, path, n_neighbors=3, save_gzip_path=None, clean_gzip=False):
        super().__init__(path, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        print("> Training")
        self.model.fit(self.ds.x_train, self.ds.y_train)

        print("Finish fit!")

        # res = self.predict()
        # print(res)
        # print("Finish predict!")

        print("Score train: ", self.model.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", self.model.score(self.ds.x_test, self.ds.y_test))

# s = TrainKNN("data/short.csv", save_gzip_path="data/knn-prepro", clean_gzip=True)
s = TrainKNN("data/projects.csv", save_gzip_path="data/knn-prepro", clean_gzip=True)
s.train()
f1 = s.evaluate()
print(f1)
