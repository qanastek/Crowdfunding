from Trainer import Trainer
from sklearn.neighbors import KNeighborsClassifier

class TrainKNN(Trainer):

    def __init__(self, path, 
                n_neighbors=3,
                save_gzip_path=None,
                clean_gzip=False,
                num_strategy=None,
                cat_strategy=None):

        super().__init__(
            path,
            save_gzip_path=save_gzip_path,
            clean_gzip=clean_gzip,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy
        )

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
s = TrainKNN("data/projects.csv", save_gzip_path="data/prepro", clean_gzip=True)

''' s = TrainKNN("data/projects.csv", 
    save_gzip_path="data/ds_med_uv", 
    clean_gzip=True,
    num_strategy="median",
    cat_strategy="unique_value")

s = TrainKNN("data/projects.csv", 
    save_gzip_path="data/ds_mean_uv", 
    clean_gzip=True,
    num_strategy="mean",
    cat_strategy="unique_value")

s = TrainKNN("data/projects.csv", 
    save_gzip_path="data/ds_med_mf", 
    clean_gzip=True,
    num_strategy="median",
    cat_strategy="most_frequent")

s = TrainKNN("data/projects.csv", 
    save_gzip_path="data/ds_mean_mf", 
    clean_gzip=True,
    num_strategy="mean",
    cat_strategy="most_frequent") '''


s.train()
f1 = s.evaluate()
print(f1)
