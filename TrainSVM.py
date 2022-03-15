from Trainer import Trainer

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

class TrainSVM(Trainer):
    """
    Trainer for the SVM
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, train_ratio=0.90, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, train_ratio=train_ratio, normalizer=normalizer)
        self.EPOCHS = 999999999

    def train(self):

        svm = LinearSVC(
            max_iter = 999,
            # max_iter = self.EPOCHS,
            # learning_rate = 0.01,
            # n_jobs = -1,
        )

        print("> START TRAINING !")
        self.model = svm.fit(self.ds.x_train, self.ds.y_train)

        res = self.predict()
        print(res)

        print("Score train: ", svm.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", svm.score(self.ds.x_test, self.ds.y_test))

# s = TrainSVM("data/short.csv", normalizer=None)
# s = TrainSVM("data/short.csv", normalizer="StandardScaler")
s = TrainSVM("data/short.csv", normalizer="MinMaxScaler")
# s = TrainSVM("data/projects.csv")
s.train()
f1 = s.evaluate()
print(f1)