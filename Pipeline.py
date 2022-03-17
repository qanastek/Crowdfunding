import os
import json
import pickle
from typing import List
from datetime import datetime

from Trainer import Trainer

from TrainSVM import TrainSVM
from TrainMLP import TrainMLP
from TrainDecisionTree import TrainDecisionTree

from sklearn.metrics import classification_report, accuracy_score

class Pipeline:
    """
    Run the benchmarks
    """

    def __init__(self, dir="benchmarks/"):

        # Models to run benchmarks
        self.models : List[Trainer] = [
            TrainSVM,
            # TrainDecisionTree,
            # TrainMLP,
        ]

        self.score_metric = "accuracy_score"
        # self.score_metric = "f1"
        self.metric_name = "Accuracy"  if self.score_metric == "accuracy_score" else "F1-Score"

        # List of results
        self.results = []

        # Output JSON
        os.makedirs(dir, exist_ok=True)
        self.date_str = datetime.now().strftime("%m/%d/%Y_%H:%M:%S").replace("/","_").replace(":","_").replace("\\","_")

        # Build directory path
        self.directory = dir if dir.endswith("/") == True else dir + "/"
        self.directory += self.date_str + "/"

        # Create output directory for the models
        os.makedirs(self.directory, exist_ok=True)

        print("Directory : ", self.directory)

        self.output_path = self.directory + "benchmark-" + self.date_str + ".json"

    def visualization(self):
        print("Visualization!")
    
    def loadModel(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def save(self, model, card):

        # Save performances
        with open(self.output_path, 'w') as o:
            json.dump(self.results, o, indent=4)

        # Get output file name
        file_name = self.directory + str(hash(str(card))) + ".pkl"

        # Save the model itself
        with open(file_name, "wb") as f:
            pickle.dump(model, f)

        return file_name

    def run(self):
        
        # For each architecture
        for arch in self.models:

            arch = arch.benchmarks()

            # Get the model
            m = arch["model"]
            name = arch["name"]

            print(f"\n\033[92m>>> {name.upper()} <<<\033[0m\n")

            # Get the hyper-parameters
            for hparams in arch["args"]:

                # Train it
                m.train(**hparams)

                # Evaluate the F1-Score
                classification_score, matrix = m.evaluate(mode="dev", method=self.score_metric)

                print("> Hyper-parameters : ", str(hparams))
                print(self.metric_name, ": ", classification_score*100, " %")
                
                # Build the model card
                model_card = {
                    "name": name,
                    "dataset": m.ds.save_gzip_path,
                    "hyper_params": hparams,
                    "score": classification_score,
                    "f1_matrix": matrix,
                    "training_date": str(datetime.now()),
                }

                # Save results and model
                model_card["model_path"] = self.save(m.model, model_card)

                self.results.append(model_card)

    def findBest(self):

        print(f"\n> TESTING HYPER-PARAMETERS FINISHED!")

        print(f"\n\033[92m>>> EVALUATE ON TEST SET <<<\033[0m")

        # Fetch test set
        dataset = self.models[0].benchmarks()["model"].ds
        x = dataset.x_test
        y = dataset.y_test
        labels = dataset.labels
        
        # For each architecture
        for arch in self.models:

            # Get attributs
            arch = arch.benchmarks()

            # Get the model name
            name = arch["name"]

            best_score = 0
            best_config = None

            # For each run
            for run in self.results:

                # Chec kif the performance have been improved
                if run["name"] == name and run["score"] > best_score:
                    best_score = run["score"]
                    best_config = run
            
            print(f"\n\033[92m>>> Best hyper-parameters for {name} <<<\033[0m")
            print(f"> \033[95m{best_config['hyper_params']}\033[0m")

            # Load best model and perform predictions on the test set
            best_model = self.loadModel(best_config["model_path"])
            preds = best_model.predict(x)

            # Evaluate Accuracy
            acc = accuracy_score(y_true=y, y_pred=preds)
            print(f"> Accuracy : {acc*100} %")

            # Evaluate F1-Score
            print(f"> Best F1-Score :")
            print(classification_report(
                y,
                preds,
                target_names = labels,
                zero_division = 1,
            ))

            print(f"> Saved at : \033[96m{best_config['model_path']}\033[0m")

p = Pipeline()
p.visualization()
p.run()
p.findBest()