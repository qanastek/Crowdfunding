from Trainer import Trainer
from sklearn.neighbors import KNeighborsClassifier

class TrainKNN(Trainer):

    def __init__(
        self,
        path,
        save_gzip_path = None,
        clean_gzip = False,
        num_strategy = None,
        cat_strategy = None,
        normalizer = None,
        normalize_currency = True,
    ):

        super().__init__(
            path,
            save_gzip_path = save_gzip_path,
            clean_gzip = clean_gzip,
            num_strategy = num_strategy,
            cat_strategy = cat_strategy,
            normalizer = normalizer,
            normalize_currency = normalize_currency
        )

    def train(self, n_neighbors=3, weights="uniform", algorithm="auto"):

        self.model = KNeighborsClassifier(
            n_neighbors = n_neighbors,
            weights = weights,
            algorithm = algorithm,
            n_jobs = -1,
        )

        print("> START TRAINING !")
        self.model.fit(self.ds.x_train, self.ds.y_train)
        print("> TRAINING FINISHED !")

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "KNN",
            "model": TrainKNN(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False,
            ),
            "args": [                
                {"n_neighbors":1, "weights":"uniform", "algorithm":"kd_tree"},
                {"n_neighbors":2, "weights":"uniform", "algorithm":"kd_tree"},
                {"n_neighbors":3, "weights":"uniform", "algorithm":"kd_tree"},
                
                {"n_neighbors":1, "weights":"distance", "algorithm":"kd_tree"},
                {"n_neighbors":2, "weights":"distance", "algorithm":"kd_tree"},
                {"n_neighbors":3, "weights":"distance", "algorithm":"kd_tree"},
            ]
        }
