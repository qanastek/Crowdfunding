import os
import json
from typing import List
from datetime import datetime

from Trainer import Trainer
from TrainSVM import TrainSVM

class Pipeline:
    """
    Run the benchmarks
    """

    def __init__(self, dir="benchmarks/"):

        # Models to run benchmarks
        self.models : List[Trainer] = [
            TrainSVM,
        ]

        # List of results
        self.results = []

        # Output JSON
        os.makedirs(dir, exist_ok=True)
        self.output_path = dir + "benchmark-" + datetime.now().strftime("%m/%d/%Y_%H:%M:%S").replace("/","_").replace(":","_").replace("\\","_") + ".json"

    def visualization(self):
        print("Visualization!")

    def save(self):

        with open(self.output_path, 'w') as o:
            json.dump(self.results, o, indent=4)

        print(f"Results file updated at {self.output_path}")

    def run(self):
        
        # For each architecture
        for arch in self.models:

            arch = arch.benchmarks()

            # Get the model
            m = arch["model"]
            name = arch["name"]

            # Get the hyper-parameters
            for hparams in arch["args"]:

                # Train it
                m.train(**hparams)

                # Evaluate the F1-Score
                f1, matrix = m.evaluate(mode="dev")
                
                self.results.append({
                    "name": name,
                    "dataset": m.ds.save_gzip_path,
                    "hyper_params": hparams,
                    "f1": f1,
                    "f1_matrix": matrix
                })

                print("F1-Score: ", f1)

                # Save benchmark current results
                self.save()

p = Pipeline()
p.visualization()
p.run()