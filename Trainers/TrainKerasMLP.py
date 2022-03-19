from Trainers.Trainer import Trainer

import numpy as np
from sklearn.metrics import classification_report

from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class TrainKerasMLP(Trainer):
    """
    Trainer for the MLP
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(
        self,
        max_epochs=1,
        batch_size=2048,
        dense_1=100,
        dense_2=256,
        drop_out=0.10,
    ):

        input_dim = len(self.ds.x_train[0])

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(input_dim,)))

        self.model.add(Dense(dense_1))
        self.model.add(Dropout(drop_out))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())

        self.model.add(Dense(dense_2))
        self.model.add(Dropout(drop_out))
        self.model.add(LeakyReLU())
        self.model.add(BatchNormalization())

        self.model.add(Dense(len(self.ds.labels), activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        print(self.model.summary())

        print("> START TRAINING !")
        self.model.fit(self.ds.x_train, self.ds.y_train, epochs=max_epochs, batch_size=batch_size, verbose=1, validation_data=(self.ds.x_dev, self.ds.y_dev))
        print("> TRAINING FINISHED !")
                
        score = self.model.evaluate(self.ds.x_test, self.ds.y_test, verbose=1)
        print(f'Test results - Loss: {score[0]} - Accuracy: {score[1]}%')
        
        predictions = self.model.predict(self.ds.x_test)
        y_preds = list(np.argmax(predictions, axis=1))
        y_test = [np.argmax(pred) for pred in self.ds.y_test]

        f1_scores = classification_report(y_test, y_preds, target_names=self.ds.labels)
        print(f1_scores)
        exit(0)

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "KerasMLP",
            "model": TrainKerasMLP(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False
            ),
            "args": [
                {"max_epochs":10, "dense_1":100, "dense_2":256, "drop_out":0.10},
                {"max_epochs":10, "dense_1":100, "dense_2":256, "drop_out":0.50},
            ]
        }
