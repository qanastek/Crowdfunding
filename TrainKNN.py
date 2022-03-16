from Trainer import Trainer
from sklearn.neighbors import KNeighborsClassifier

class TrainKNN(Trainer):

    def __init__(self, path, 
                n_neighbors=3,
                save_gzip_path=None,
                clean_gzip=False,
                num_strategy=None,
                cat_strategy=None,
                normalizer="StandardScaler",
                normalize_currency = True):

        super().__init__(path,
            save_gzip_path=save_gzip_path,
            clean_gzip=clean_gzip,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            normalizer=normalizer,
            normalize_currency = normalize_currency)

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        print("> Training")
        self.model.fit(self.ds.x_train, self.ds.y_train)
        print("Finish fit!")
        return self.model.score(self.ds.x_train, self.ds.y_train)

def test():

    ''' 
    Add test() at the end of this file and run ─▶ python3 TrainKNN.py :
    Out :
        > Data loaded - DONE!
        > Training
        Finish fit!
        Training-score : 0.78190961948805
        Test F-Score :               precision    recall  f1-score   support

                   0       0.60      0.60      0.60     21145
                   1       0.59      0.60      0.59     20877

            accuracy                           0.60     42022
           macro avg       0.60      0.60      0.60     42022
        weighted avg       0.60      0.60      0.60     42022 
    '''


    s = TrainKNN("data/projects.csv", save_gzip_path="data/ds_mean_uv")
    print("Training-score : " + str(s.train()))
    print("Test F-Score : " + str(s.evaluate()))