from abc import abstractmethod

from Dataset import Dataset
from sklearn.metrics import classification_report

class Trainer:
    """
    Basic trainer for sklearn models
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, train_ratio=0.60, dev_ratio=0.20, test_ratio=0.20, normalizer="StandardScaler", normalize_currency=True):
        self.ds = Dataset(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio, normalizer=normalizer, normalize_currency=normalize_currency)
        self.model = None

    @abstractmethod
    def train(self):
        print("Please implement the train function first!")
        self.model = None
        raise NotImplementedError

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, mode="dev"):

        if mode == "dev":
            x, y = self.ds.x_dev, self.ds.y_dev
        elif mode == "test":
            x, y = self.ds.x_test, self.ds.y_test
        else:
            print("Unknown mode!")
            exit(0)

        preds = self.predict(x)
        f1_score = classification_report(y, preds, target_names=self.ds.labels, zero_division=1)
        return f1_score