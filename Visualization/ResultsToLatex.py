import re, os, sys, json

class ResultsToLatex:

    def __init__(self, output_dir="../benchmarks"):

        self.output_dir = output_dir

        usage = """- Usage : 
        \n\t - python3 ResultsToLatex.py test => Shows only test output
        \n\t - python3 ResultsToLatex.py dev => Shows only dev output
        \n\t - python3 ResultsToLatex.py => Show both\n"""


        argLim = 'all'
        if ( len(sys.argv) >= 2 ):
            argLim = sys.argv[1]
            if 'h' in argLim or 'u' in argLim:
                print(usage)
                exit()  

        models = {}
        iModel = 'test'
        models[iModel] = {}
        all_res = []

        for root, dirs, files in os.walk(self.output_dir):
            
            for file in files :

                path_file = os.path.join(root, file)

                if '.txt' in file: # Test

                    model_name = file.split("-")[1].split(".")[0]
                    if ( model_name not in models[iModel]  ):
                        models[iModel][model_name] = {}
                        models[iModel][model_name]['score'] = 0
                    file_content = open(path_file,"r").read()
                    f1_score = file_content.split("\n")[11]        
                    f1_score = re.sub("\s+"," ",f1_score)
                    _, __, prec, recc, f1, support = re.sub("\s+", " ", f1_score).split(" ")

                    accuracy = file_content.split("\n")[9]     
                    accuracy = re.sub("\s+", " ", accuracy).split(" ")[-2]

                    new_score = float(f1)*100

                    if new_score > models[iModel][model_name]["score"]:
                        models[iModel][model_name]["score"] = new_score
                        models[iModel][model_name]["f1_score"] = {
                            "precision": float(prec)*100,
                            "recall": float(recc)*100,
                            "accuracy": float(accuracy)*100,
                            "f1_score": float(f1)*100,
                    }

                if '.json' in file: # Dev
                    r = json.load(open(path_file, 'r'))
                    all_res.extend(r)

        iModel = 'dev'
        models[iModel]={}

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
            models[iModel][model_name] = {}
            models[iModel][model_name]['score'] = 0

            if new_score > models[iModel][model_name]["score"]:
                    
                models[iModel][model_name]["score"] = new_score
                models[iModel][model_name]["f1_score"] = {
                    "precision": float(prec)*100,
                    "recall": float(recc)*100,
                    "accuracy": float(accuracy)*100,
                    "f1_score": float(f1)*100,
                }

    def generateLatex(models):

        for corp, corpus in zip(models, ['test','dev']):

            if argLim == corp or argLim == 'all' :

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

                for mod in models[corp]:
                    a = "%.2f" % models[corp][mod]["f1_score"]["accuracy"]
                    p = "%.2f" % models[corp][mod]["f1_score"]["precision"]
                    r = "%.2f" % models[corp][mod]["f1_score"]["recall"]
                    f = "%.2f" % models[corp][mod]["f1_score"]["f1_score"]
                    print("\t" +str(mod), "&", a, "\\% &", p, "\\% &", r, "\\% &", f, "\\% \\\\")

                footer = """
                \\hline
                \\end{tabular}
                % }
                \\caption{Résultats """+ corpus +""".}
                \\label{tab:"""+corpus+"""Results}
                \\end{table}
                """
                print(footer)