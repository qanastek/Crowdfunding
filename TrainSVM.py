from Trainer import Trainer

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

class TrainSVM(Trainer):
    """
    Trainer for the SVM
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)
        self.penalities = ["l2","l1"]

    def train(self, epochs=15, penalty="l2", loss="squared_hinge", dual=True, C=1.0, tol=1e-4):

        self.model = LinearSVC(
            max_iter = epochs,
            penalty = penalty,
            loss = loss,
            dual = dual,
            C = C,
            tol = tol,
        )

        self.model.fit(self.ds.x_train, self.ds.y_train)

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "SVM",
            "model": TrainSVM(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False,
            ),
            "args": [
                {"epochs":15, "penalty":"l1", "loss":"squared_hinge", "dual":False, "C":1.0, "tol":1e-4},
                {"epochs":15, "penalty":"l2", "loss":"squared_hinge", "dual":False, "C":1.0, "tol":1e-4},
            ]
        }
