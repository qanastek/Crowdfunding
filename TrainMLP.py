from Trainer import Trainer

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class TrainMLP(Trainer):
    """
    Trainer for the MLP
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)
        self.EPOCHS = 999999999

    def train(self):

        m = MLPClassifier(
            random_state=1,
            max_iter = 1,
            # learning_rate = 0.01,
        )

        print("> START TRAINING !")
        self.model = m.fit(self.ds.x_train, self.ds.y_train)

        print("Score train: ", m.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", m.score(self.ds.x_test, self.ds.y_test))

s = TrainMLP("data/projects.csv", normalizer="StandardScaler")
s.train()
f1 = s.evaluate()
print(f1)