from Trainers.Trainer import Trainer

from sklearn.neural_network import MLPClassifier

class TrainMLP(Trainer):
    """
    Trainer for the MLP
    """

    def __init__(self, path, shuffle=True, seed=0, save_gzip_path=None, clean_gzip=False, normalizer="StandardScaler"):
        super().__init__(path, shuffle=shuffle, seed=seed, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip, normalizer=normalizer)

    def train(
        self,
        epochs=15,
        activation="relu",
        solver="adam",
        learning_rate="constant",
        learning_rate_init=0.001,
        early_stopping=True,
        batch_size=4096,
        hidden_layer_sizes=100,
    ):

        self.model = MLPClassifier(
            random_state = 1,
            max_iter = epochs,
            activation = activation,
            solver = solver,
            learning_rate = learning_rate,
            learning_rate_init = learning_rate_init,
            early_stopping = early_stopping,
            batch_size = batch_size,
            hidden_layer_sizes = hidden_layer_sizes,
        )

        self.model.fit(self.ds.x_train, self.ds.y_train)

    @staticmethod
    def benchmarks():
        data_path = "data/StandardScaler_only"
        return {
            "name": "MLP",
            "model": TrainMLP(
                "data/projects.csv",
                normalizer="StandardScaler",
                save_gzip_path=data_path,
                clean_gzip=False
            ),
            "args": [

                {"epochs":500, "activation":"relu", "solver":"sgd", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":50},
                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":50},
                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":100},

                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":1.0, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.1, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.01, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":250},

                {"epochs":500, "activation":"relu", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":500},



                {"epochs":500, "activation":"tanh", "solver":"sgd", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":50},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":50},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":100},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":250},
                
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":1.0, "early_stopping":True, "hidden_layer_sizes":500},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.1, "early_stopping":True, "hidden_layer_sizes":500},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.01, "early_stopping":True, "hidden_layer_sizes":500},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":500},
                {"epochs":500, "activation":"tanh", "solver":"adam", "learning_rate":"adaptive", "learning_rate_init":0.0001, "early_stopping":True, "hidden_layer_sizes":500},



                {"epochs":1500, "activation":"tanh", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":50},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":100},

                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":2.0, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":1.50, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":1.25, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":1.0, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.1, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.01, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":250},
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.0001, "early_stopping":True, "hidden_layer_sizes":250},
                
                {"epochs":1500, "activation":"relu", "solver":"lbfgs", "learning_rate":"adaptive", "learning_rate_init":0.001, "early_stopping":True, "hidden_layer_sizes":500},
            ]
        }
