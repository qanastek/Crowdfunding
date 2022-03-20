import re

DIR_1 = "../benchmarks/03_19_2022_00_33_54/"
json_1 = DIR_1 + "benchmark-03_19_2022_00_33_54.json"

DIR_2 = "../benchmarks/03_19_2022_01_06_01/"
json_2 = DIR_2 + "benchmark-03_19_2022_01_06_01.json"

models_files = [
    "best_model-DecisionTree.txt",
    "best_model-KNN.txt",
    "best_model-MLP.txt",
    "best_model-NaiveBayes.txt",
    "best_model-SVM.txt",
]

dirs = [DIR_1,DIR_2]
jsons = [json_1,json_2]

models = {
    "DecisionTree": {
        "score": 0.0,
        "f1_score": None,
    },
    "KNN": {
        "score": 0.0,
        "f1_score": None,
    },
    "MLP": {
        "score": 0.0,
        "f1_score": None,
    },
    "NaiveBayes": {
        "score": 0.0,
        "f1_score": None,
    },
    "SVM": {
        "score": 0.0,
        "f1_score": None,
    },
}

for dir, j in zip(dirs, jsons):

    for model_file in models_files:

        model_name = model_file.split("-")[1].split(".")[0]
        path_file = dir + model_file

        file_content = open(path_file,"r").read()

        f1_score = file_content.split("\n")[11]        
        f1_score = re.sub("\s+"," ",f1_score)
        _, __, prec, recc, f1, support = re.sub("\s+", " ", f1_score).split(" ")

        accuracy = file_content.split("\n")[9]     
        accuracy = re.sub("\s+", " ", accuracy).split(" ")[-2]

        new_score = float(f1)*100

        if new_score > models[model_name]["score"]:
            models[model_name]["score"] = new_score
            models[model_name]["f1_score"] = {
                "precision": float(prec)*100,
                "recall": float(recc)*100,
                "accuracy": float(accuracy)*100,
                "f1_score": float(f1)*100,
            }

        # print(model_file)
        # print(model_name)
        # print(dir)
        # print(j)
        # print()

print(models)

header = """
\\begin{table}[htb!]
\\centering
% \\resizebox{\\linewidth}{!}{%
\\rowcolors{1}{fgVeryLightRed}{}
\\begin{tabular}{ccccc}
\\rowcolor{fgLightRed}
\\hline
\\textbf{Model} & \\textbf{Taux d'erreur} & \\textbf{Précision} & \\textbf{Rappel} & \\textbf{F1\\_Score} \\\\ \\hline
"""

print(header)

for m in models:

    a = "%.2f" % models[m]["f1_score"]["accuracy"]
    p = "%.2f" % models[m]["f1_score"]["precision"]
    r = "%.2f" % models[m]["f1_score"]["recall"]
    f = "%.2f" % models[m]["f1_score"]["f1_score"]

    print(m, "&", a, "\\% &", p, "\\% &", r, "\\% &", f, "\\% \\\\")

footer = """
\\hline
\\end{tabular}
% }
\\caption{Résultats sur Test.}
\\label{tab:DevResults}
\\end{table}
"""
print(footer)