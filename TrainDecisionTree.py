from Trainer import Trainer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class TrainDecisionTree(Trainer):
    """
    Trainer for the DecisionTreeClassifier
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(self, depth=10, criterion="gini"):

        self.model = DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
        )

        print("> START TRAINING !")
        self.model.fit(self.ds.x_train, self.ds.y_train)
        print("> TRAINING FINISHED !")

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "DecisionTree",
            "model": TrainDecisionTree(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False
            ),
            "args": [

                {"depth":1, "criterion":"gini"},
                {"depth":3, "criterion":"gini"},
                {"depth":5, "criterion":"gini"},
                {"depth":10, "criterion":"gini"},

                {"depth":1, "criterion":"entropy"},
                {"depth":3, "criterion":"entropy"},
                {"depth":5, "criterion":"entropy"},
                {"depth":10, "criterion":"entropy"},
            ]
        }
