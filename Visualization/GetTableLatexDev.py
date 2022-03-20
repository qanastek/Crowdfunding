import re
import json

all_res = []

DIR_1 = "../benchmarks/03_19_2022_00_33_54/"
json_1 = DIR_1 + "benchmark-03_19_2022_00_33_54.json"

with open(json_1, 'r') as f:
    r = json.load(f)
    all_res.extend(r)

print(len(all_res))

DIR_2 = "../benchmarks/03_19_2022_01_06_01/"
json_2 = DIR_2 + "benchmark-03_19_2022_01_06_01.json"

with open(json_2, 'r') as f:
    r = json.load(f)
    all_res.extend(r)

print(len(all_res))

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

for e in all_res:

    model_name = e["name"]
    content = e["f1_matrix"]
    
    f1_score = content.split("\n")[7]
    f1_score = re.sub("\s+"," ",f1_score)
    _, __, prec, recc, f1, support = re.sub("\s+", " ", f1_score).split(" ")

    accuracy = content.split("\n")[5]     
    accuracy = re.sub("\s+", " ", accuracy).split(" ")[-2]

    score = f1

    new_score = float(f1)*100

    if new_score > models[model_name]["score"]:
        models[model_name]["score"] = new_score
        models[model_name]["f1_score"] = {
            "precision": float(prec)*100,
            "recall": float(recc)*100,
            "accuracy": float(accuracy)*100,
            "f1_score": float(f1)*100,
        }

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
\\caption{Résultats sur Dev.}
\\label{tab:DevResults}
\\end{table}
"""
print(footer)