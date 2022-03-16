from Trainer import Trainer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class TrainDecisionTree(Trainer):
    """
    Trainer for the DecisionTreeClassifier
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer, train_ratio=0.75)
        self.EPOCHS = 999999999

    def train(self):

        classifier = DecisionTreeClassifier(max_depth=10)
        # classifier = DecisionTreeClassifier()

        print("> START TRAINING !")
        self.model = classifier.fit(self.ds.x_train, self.ds.y_train)

        print("Score train: ", classifier.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", classifier.score(self.ds.x_test, self.ds.y_test))

# s = TrainSVM("data/short.csv", normalizer="StandardScaler")
# s = TrainSVM("data/short.csv", normalizer="MinMaxScaler")
# s = TrainSVM("data/short.csv", normalizer=None)
s = TrainDecisionTree("data/projects.csv", normalizer="StandardScaler")
s.train()
f1 = s.evaluate()
print(f1)