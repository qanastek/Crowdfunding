import re, os, json

def argsToLatex(corpus_sets):

    all_latex = {
        "train": "",
        "dev": "",
        "test": "",
    }

    # For each corpus sets
    for corpus_set in corpus_sets:

        # List of models for this corpus set
        set_models = corpus_sets[corpus_set]
            
        header = """
        \\begin{table}[htb!]
        \\centering
        % \\resizebox{\\linewidth}{!}{%
        \\rowcolors{1}{fgVeryLightRed}{}
        \\begin{tabular}{ccccc}
        \\rowcolor{fgLightRed}
        \\hline
        \\textbf{Model} & \\textbf{Taux d'erreur} & \\textbf{Précision} & \\textbf{Rappel} & \\textbf{F1\\_Score} \\\\ \\hline\n"""

        body = ""

        # Get the models
        for i, m in enumerate(set_models):

            metrics = set_models[m]["f1_score"]

            a = "%.2f" % metrics["accuracy"]
            p = "%.2f" % metrics["precision"]
            r = "%.2f" % metrics["recall"]
            f = "%.2f" % metrics["f1_score"]
            body += "\t" + str(m) + " & " + a + " \\% & " + p + " \\% & " + r + " \\% & " + f + " \\% \\\\"
            
            if i != len(set_models)-1:
                body += "\n"

        footer = """ 
        \\hline\\end{tabular}
        % }
        \\caption{Résultats """ + corpus_set + """.}
        \\label{tab:""" + corpus_set + """Results}
        \\end{table}
        """
        
        all_latex[corpus_set] = header + body + footer

    return all_latex

class ResultsToLatex:

    def __init__(self, input_dir, save_dir=None):

        # Input / Output dirs
        self.input_dir = input_dir
        self.save_dir = save_dir

        self.models = {
            'train': {},
            'dev': {},
            'test': {},
        }

        # Load .json and .txt
        self.loadFiles()

        # Generate latex files
        self.generateLatex()

    def F1MatrixToDict(self, content, f1_idx=11, acc_idx=9):
        
        # Get the f1 score metrics 
        f1_score = content.split("\n")[f1_idx]        
        f1_score = re.sub("\s+"," ",f1_score)
        _, __, prec, recc, f1, support = re.sub("\s+", " ", f1_score).split(" ")

        # Get the accuracy
        accuracy = content.split("\n")[acc_idx]     
        accuracy = re.sub("\s+", " ", accuracy).split(" ")[-2]

        # Get the comparison score
        new_score = float(f1)*100

        f1_score = {
            "precision": float(prec)*100,
            "recall": float(recc)*100,
            "accuracy": float(accuracy)*100,
            "f1_score": float(f1)*100,
        }

        return new_score, f1_score

    def loadFiles(self):
        
        # For each directory
        for root, dirs, files in os.walk(self.input_dir):
            
            # For each file in the directory
            for file in files:

                # Path file
                path_file = os.path.join(root, file)

                # Test
                if '.txt' in file and "best_model-" in file:

                    corpus_set = "test"

                    # Get model name
                    model_name = file.split("-")[1].split(".")[0]

                    # Add the model if isn't registered yet
                    if model_name not in self.models[corpus_set]:
                        self.models[corpus_set][model_name] = {}
                        self.models[corpus_set][model_name]['score'] = 0

                    model_perfs = self.models[corpus_set][model_name]

                    # Read the text file
                    file_content = open(path_file,"r").read()

                    # Parse F1-Score matrix
                    new_score, f1_score = self.F1MatrixToDict(file_content, f1_idx=11, acc_idx=9)

                    # If better
                    if new_score > model_perfs["score"]:

                        # Keep the score
                        model_perfs["score"] = new_score

                        # Keep the f1 score metrics
                        model_perfs["f1_score"] = f1_score

                # Dev
                if '.json' in file:

                    if '.train' in file:
                        corpus_set = "train"
                    else:
                        corpus_set = "dev"

                    # Load and merge the json
                    runs = json.load(open(path_file, 'r'))

                    for r in runs:

                        model_name = r["name"]

                        # Add the model if isn't registered yet
                        if model_name not in self.models[corpus_set]:
                            self.models[corpus_set][model_name] = {}
                            self.models[corpus_set][model_name]['score'] = 0
                            
                        model_perfs = self.models[corpus_set][model_name]
                            
                        # Parse F1-Score matrix
                        new_score, f1_score = self.F1MatrixToDict(r["f1_matrix"], f1_idx=7, acc_idx=5)

                        # If better
                        if new_score > model_perfs["score"]:

                            # Keep the score
                            model_perfs["score"] = new_score

                            # Keep the f1 score metrics
                            model_perfs["f1_score"] = f1_score

    def generateLatex(self):

        print("> Generate Latex")

        all_latex = argsToLatex(self.models)

        for latex in all_latex:

            if self.save_dir == None:
                print(all_latex[latex])
            else:
                output_file = self.save_dir + "latex_" + latex + ".txt"
                print("> Saved in : ", output_file)
                of = open(output_file, "w")
                of.write(all_latex[latex])
                of.close()

if __name__ == '__main__':

    ResultsToLatex(input_dir="../benchmarks", save_dir="./")