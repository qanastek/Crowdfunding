from Trainer import Trainer
from sklearn.neighbors import KNeighborsClassifier

class TrainKNN(Trainer):

    def __init__(self, path, 
                n_neighbors=3,
                save_gzip_path=None,
                clean_gzip=False,
                num_strategy=None,
                cat_strategy=None,
                normalizer="StandardScaler",
                normalize_currency = True):

        super().__init__(path,
            save_gzip_path=save_gzip_path,
            clean_gzip=clean_gzip,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            normalizer=normalizer,
            normalize_currency = normalize_currency)

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        print("> Training")
        self.model.fit(self.ds.x_train, self.ds.y_train)
        print("Finish fit!")
        print("Score train: ", self.model.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", self.model.score(self.ds.x_test, self.ds.y_test))

def test():

    s = TrainKNN("data/projects.csv", save_gzip_path="data/ds_mean_uv")
    s.train()
    f1 = s.evaluate()
    print(f1)