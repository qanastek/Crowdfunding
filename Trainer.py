from abc import abstractmethod

from Dataset import Dataset
from sklearn.metrics import classification_report

class Trainer:
    """
    Basic trainer for sklearn models
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, train_ratio=0.99, normalizer="StandardScaler"):
        self.ds = Dataset(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, train_ratio=train_ratio, normalizer=normalizer)
        self.model = None

    @abstractmethod
    def train(self):
        print("Please implement the train function first!")
        self.model = None
        raise NotImplementedError

    def predict(self):
        return self.model.predict(self.ds.x_test)

    def evaluate(self):
        preds = self.predict()
        f1_score = classification_report(self.ds.y_test, preds, target_names=self.ds.labels, zero_division=1)
        return f1_score