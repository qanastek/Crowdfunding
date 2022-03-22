from Trainers.Trainer import Trainer

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
                {"var_smoothing":1e-1},
                {"var_smoothing":1e-2},
                {"var_smoothing":1e-3},
                {"var_smoothing":1e-4},
                {"var_smoothing":1e-5},
                {"var_smoothing":1e-6},
                {"var_smoothing":1e-7},
                {"var_smoothing":1e-8},
                
                {"var_smoothing":1e-9},
                
                {"var_smoothing":1e-10},
                {"var_smoothing":1e-11},
                {"var_smoothing":1e-12},
                {"var_smoothing":1e-13},
                {"var_smoothing":1e-14},
            ]
        }
