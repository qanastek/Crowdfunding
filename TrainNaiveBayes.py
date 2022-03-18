from Trainer import Trainer
from sklearn.naive_bayes import GaussianNB

class TrainNaiveBayes(Trainer):
    """
    Trainer for the Naive Bayes
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(self, var_smoothing=1e-9):

        self.model = GaussianNB(
            var_smoothing = var_smoothing,
        )

        self.model.fit(self.ds.x_train, self.ds.y_train)

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "NaiveBayes",
            "model": TrainNaiveBayes(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False,
            ),
            "args": [
                {"var_smoothing":1e-9},
            ]
        }
