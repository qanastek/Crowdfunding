from Trainer import Trainer

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class TrainMLP(Trainer):
    """
    Trainer for the MLP
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(
        self,
        epochs=15,
        activation="relu",
        solver="adam",
        learning_rate="constant",
        learning_rate_init=0.001,
        early_stopping=True,
    ):

        self.model = MLPClassifier(
            random_state = 1,
            max_iter = epochs,
            activation = activation,
            solver = solver,
            learning_rate = learning_rate,
            learning_rate_init = learning_rate_init,
            early_stopping = early_stopping,
        )

        self.model.fit(self.ds.x_train, self.ds.y_train)

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "MLP",
            "model": TrainMLP(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False
            ),
            "args": [

                {"epochs":15, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True},
                {"epochs":15, "activation":"relu", "solver":"sgd", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True},

                {"epochs":15, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True},
                {"epochs":15, "activation":"tanh", "solver":"sgd", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True},
            ]
        }
