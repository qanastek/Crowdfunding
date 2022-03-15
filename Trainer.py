from abc import abstractmethod

from Dataset import Dataset
from sklearn.metrics import classification_report

class Trainer:
    """
    Basic trainer for sklearn models
    """

    def __init__(self, path, save_gzip_path=None):
        self.ds = Dataset(path, save_gzip_path=save_gzip_path)
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
        f1_score = classification_report(self.ds.y_test, preds, target_names=self.ds.labels)
        return f1_score