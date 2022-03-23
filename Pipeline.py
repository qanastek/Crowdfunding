import os
import sys
import json
import pickle
from typing import List
from datetime import datetime

from sklearn.metrics import classification_report, accuracy_score

from Trainers.Trainer import Trainer

from Trainers.TrainSVM import TrainSVM
from Trainers.TrainMLP import TrainMLP
from Trainers.TrainKNN import TrainKNN
from Trainers.TrainKerasMLP import TrainKerasMLP
from Trainers.TrainNaiveBayes import TrainNaiveBayes
from Trainers.TrainDecisionTree import TrainDecisionTree

from Visualization.DataAnalysis import DataAnalysis
from Visualization.ResultsToLatex import ResultsToLatex

class Pipeline:
    """
    Run the benchmarks
    """

    def __init__(self, dir="benchmarks/", with_visualization=False):

        # Models to run benchmarks
        self.models : List[Trainer] = [
            TrainNaiveBayes,
            # TrainDecisionTree,
            # TrainSVM,
            # TrainMLP,
            # TrainKNN,
        ]

        self.score_metric = "accuracy_score"
        # self.score_metric = "f1"
        self.metric_name = "Accuracy"  if self.score_metric == "accuracy_score" else "F1-Score"

        # List of results
        self.results = []
        self.results_on_train = []

        # Output JSON
        os.makedirs(dir, exist_ok=True)
        self.date_str = datetime.now().strftime("%m/%d/%Y_%H:%M:%S").replace("/","_").replace(":","_").replace("\\","_")

        # Build directory path
        self.directory = dir if dir.endswith("/") == True else dir + "/"
        self.directory += self.date_str + "/"

        # Create output directory for the models
        os.makedirs(self.directory, exist_ok=True)

        print("Directory : ", self.directory)
        base = self.directory + "benchmark-" + self.date_str
        self.output_path = base + ".json"
        self.output_path_train = base + ".train.json"

        # Run the visualization in a background thread
        self.dataVisualization(with_visualization)

    def dataVisualization(self, with_visualization=False, with_pair_grid=False):
        """
        Run the visualization in a background thread
        """

        def run(with_visualization, with_pair_grid):

            if with_visualization == False:
                return

            print("\n" + "#"*78 + "\n" + " "*34 + "[DATASET]\n" + "#"*78)
            data = DataAnalysis(DataAnalysis.DATA_PROJECTS_FILE_H5)

            print("\n\n" + "#"*78 + "\n" + " "*34 + "[ANALYSIS]\n" + "#"*78)

            data.print_statistics()
            data.build_plots_numerical('raw_data',with_pair_grid)
            data.build_plots_categorial('raw_data')

            data.clean_data()
            
            data.print_statistics()
            data.build_plots_numerical('clean_data',with_pair_grid)
            data.build_plots_categorial('clean_data')

            data.sample_data() # Downsampling the data
            
            data.print_statistics()
            data.build_plots_numerical('sampled_data',with_pair_grid)
            data.build_plots_categorial('sampled_data')

        run(with_visualization, with_pair_grid)

    def loadModel(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def save(self, model, card):

        # Get output file name
        file_name = self.directory + str(hash(str(card))) + ".pkl"

        # Save the model itself
        with open(file_name, "wb") as f:
            pickle.dump(model, f)

        return file_name

    def evaluate(self, model_name, hparams, model, mode):

        # Evaluate the F1-Score
        classification_score, matrix = model.evaluate(mode=mode, method=self.score_metric)

        print(f"> {self.metric_name} of {classification_score*100} % using Hyper-parameters : {str(hparams)}")
        
        # Build the model card
        model_card = {
            "name": model_name,
            "dataset": model.ds.save_gzip_path,
            "hyper_params": hparams,
            "score": classification_score,
            "f1_matrix": matrix,
            "training_date": str(datetime.now()),
        }

        # Save results and model
        model_card["model_path"] = self.save(model.model, model_card)

        return model_card

    def TrainEvaluate(self, m, hparams, name):

        # Train it
        m.train(**hparams)

        # Evaluate on Train
        model_card = self.evaluate(model_name=name, hparams=hparams, model=m, mode="train")
        self.results_on_train.append(model_card)

        # Evaluate on Dev
        model_card = self.evaluate(model_name=name, hparams=hparams, model=m, mode="dev")
        self.results.append(model_card)

    def gridSearch(self):
        """
        Brute force grid search using hand defined hyper-parameters
        """

        print("\n\n" + "#"*78 + "\n" + " "*34 + "[TRAIN MODELS]\n" + "#"*78)
        
        # For each architecture
        for arch in self.models:

            arch = arch.benchmarks()

            # Get the model
            m = arch["model"]
            name = arch["name"]

            print(f"\n\033[92m>>> {name.upper()} <<<\033[0m\n")

            # jobs = []

            # Get the hyper-parameters
            for hparams in arch["args"]:

                self.TrainEvaluate(m, hparams, name)
                # jobs.append([m, hparams, name])

            # pqdm(jobs, self.TrainEvaluate, n_jobs=self.threads, argument_type='args')

        # Save performances on train
        with open(self.output_path_train, 'w') as o:
            json.dump(self.results_on_train, o, indent=4)

        # Save performances on dev
        with open(self.output_path, 'w') as o:
            json.dump(self.results, o, indent=4)

        print(f"\n> TESTING HYPER-PARAMETERS FINISHED!")

    def findBest(self):

        print("\n\n" + "#"*78 + "\n" + " "*24 + "\033[92m>>> EVALUATE ON TEST SET <<<\033[0m\n" + "#"*78)

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

            f1 = classification_report(
                y,
                preds,
                target_names = labels,
                zero_division = 1,
                digits = 4,
            )

            # Evaluate F1-Score
            print(f"> Best F1-Score :")
            print(f1)

            # Save best model F1-Score
            file_out = open(self.directory + "best_model-" + name + ".txt", "w")
            file_out.write(f"Accuracy of {acc*100} % \n\n {str(best_config['hyper_params'])} \n\n {str(f1)}")
            file_out.close()

            print(f"> Saved at : \033[96m{best_config['model_path']}\033[0m")

p = Pipeline(with_visualization=False, with_pair_grid=False)
p.gridSearch()
p.findBest()
ResultsToLatex(input_dir="./benchmarks", save_dir="./Visualization/")