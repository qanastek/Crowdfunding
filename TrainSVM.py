from Trainer import Trainer

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

class TrainSVM(Trainer):
    """
    Trainer for the SVM
    """

    def __init__(self, path):
        super().__init__(path)
        self.EPOCHS = 999999999

    def train(self):

        svm = LinearSVC(
            max_iter = 999,
            # max_iter = self.EPOCHS,
            # learning_rate = 0.01,
            # n_jobs = -1,
        )

        print("> START TRAINING !")
        self.model = svm.fit(self.ds.x_train, self.ds.y_train)

        print("Score train: ", svm.score(self.ds.x_train, self.ds.y_train))
        print("Score test: ", svm.score(self.ds.x_test, self.ds.y_test))

s = TrainSVM("data/short.csv")
# s = TrainSVM("data/projects.csv")
s.train()
f1 = s.evaluate()
print(f1)