from sklearn.metrics import classification_report, precision_recall_fscore_support
from numpyencoder import NumpyEncoder
import argparse
from pathlib import Path
import pandas as pd
import json
import typer

app = typer.Typer()

@app.command()
def comptue_score(
    predict_file: Path = typer.Option(..., help="Path to prediction file"),
    answer_file: Path = typer.Option(..., help="Path to answer file"),
):
    predict_table = pd.read_csv(predict_file, delimiter="\t", index_col="index")
    prediction = predict_table["prediction"].to_list()

    with open(answer_file, 'r', encoding='utf-8') as infile:
        answer = json.load(infile)
    labels = [a["label"] for a in answer]
    
    # comptue score
    report = classification_report(
        y_true=labels,
        y_pred=prediction,
        digits=6,
    )
    print(report)

    # get micro f1 score
    scores = precision_recall_fscore_support(
        y_true=labels,
        y_pred=prediction,
        average="micro",
    )
    print("Micro F1 Score", scores[2])

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-prediction", type=Path, required=True)
#     parser.add_argument("-answer", type=Path, required=True)
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     comptue_score(
#         predict_file=args.prediction,
#         answer_file=args.answer,
#     )

if __name__ == "__main__":
    app()
