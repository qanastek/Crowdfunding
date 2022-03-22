from Trainers.Trainer import Trainer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class TrainDecisionTree(Trainer):
    """
    Trainer for the Decision Tree
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(self, depth=10, criterion="gini"):

        self.model = DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
        )
        # self.model = RandomForestClassifier(
        #     max_depth=depth,
        #     criterion=criterion,
        #     n_jobs=-1
        # )

        self.model.fit(self.ds.x_train, self.ds.y_train)

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
                {"depth":20, "criterion":"gini"},
                {"depth":30, "criterion":"gini"},
                {"depth":40, "criterion":"gini"},
                {"depth":50, "criterion":"gini"},
                {"depth":100, "criterion":"gini"},
                {"depth":None, "criterion":"gini"},

                {"depth":1, "criterion":"entropy"},
                {"depth":3, "criterion":"entropy"},
                {"depth":5, "criterion":"entropy"},
                {"depth":6, "criterion":"entropy"},
                {"depth":7, "criterion":"entropy"},
                {"depth":8, "criterion":"entropy"},
                {"depth":9, "criterion":"entropy"},
                {"depth":10, "criterion":"entropy"},
                {"depth":20, "criterion":"entropy"},
                {"depth":30, "criterion":"entropy"},
                {"depth":40, "criterion":"entropy"},
                {"depth":50, "criterion":"entropy"},
                {"depth":100, "criterion":"entropy"},
                {"depth":None, "criterion":"entropy"},
            ]
        }
