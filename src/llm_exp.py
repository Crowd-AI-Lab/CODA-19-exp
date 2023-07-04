import torch
import os
import json
import re
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from typing import Optional
from pprint import pprint
import typer
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from typing import Dict, List
from llm import LLM, ChatGPT, GPT3

app = typer.Typer()

@app.command()
@torch.no_grad()
def exp(
    model_path: Optional[str] = typer.Option(default=None),
    model_type: str = typer.Option(..., help="Model type: gpt3, chatgpt, gpt4, llama, mpt, dolly"),
    test_file: str = typer.Option(...),
    prompt_file: str = typer.Option(...),
    output_folder: str = typer.Option(...),
):
    # load model
    assert model_type in {"gpt3", "chatgpt", "gpt4", "llama", "mpt", "dolly"}
    if model_type == "gpt3":
        model = GPT3()
    elif model_type == "chatgpt":
        model = ChatGPT(model="gpt-3.5-turbo")
    elif model_type == "gpt4":
        model = ChatGPT(model="gpt4")
    else:
        assert model_path is not None
        model = LLM(model_type=model_type, model_name_or_path=model_path)

    # load data (in json)
    with open(test_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        data = data[:]

    # load prompt
    with open(prompt_file, 'r', encoding='utf-8') as infile:
        prompt_template = infile.read()

    # create output folder
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # iterate over data
    for index, sample in tqdm(
        enumerate(data),
        total=len(data),
        desc="Inferencing"
    ):
        # skip if exists
        if Path(output_folder, f"{index}.json").exists():
            continue

        # build prompt
        if model_type in {"chatgpt", "gpt4"}:
            prompt = [
                {
                    "role": "user", 
                    "content": prompt_template.format(sample["text"]),
                },
            ]
        else:
            prompt = prompt_template.format(sample["text"])

        # generate
        response = model.generate(prompt)
        response["prompt"] = prompt

        # save output
        output_file = Path(output_folder, f"{index}.json")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(response, outfile, indent=2)


class TextExtractor:
    def __init__(self, model_type: str):
        self.model_type = model_type

        assert model_type in {"gpt3", "chatgpt", "gpt4", "llama", "mpt", "dolly"}

        if model_type == "gpt3":
            self.extractor = self.gpt3
        elif model_type == "chatgpt" or model_type == "gpt4":
            self.extractor = self.chatgpt
        else:
            self.extractor = self.openllm
    
    def __call__(self, response: Dict) -> str:
        return self.extractor(response)

    def gpt3(self, response: Dict) -> str:
        response = response["choices"][0]["text"].strip().lower()
        response = (
            f"{'[' if not response.startswith('[') else ''}"
            f"{response}"
            f"{']' if not response.endswith(']') else ''}"
        )
        return response
    
    def chatgpt(self, response: Dict) -> str:
        response = response["choices"][0]["message"]["content"].strip().lower()
        response = (
            f"{'[' if not response.startswith('[') else ''}"
            f"{response}"
            f"{']' if not response.endswith(']') else ''}"
        )

        return response
    
    def openllm(self, response: Dict) -> str:
        prompt = response["text"]
        response = response["response"]
        return response[len(prompt)-2:].strip().lower()

@app.command()
def compute_score(
    test_file: Path = typer.Option(..., help="Path to test file"),
    output_folder: Path = typer.Option(..., help="Folder containing the output json files"),
    model_type: str = typer.Option(..., help="Model type: gpt3, chatgpt, gpt4, llama, mpt, dolly"),
):
    # load data
    with open(test_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        data = data[:]

    # get unique labels from data
    label_list = {sample["label"] for sample in data}
    print("Label List", label_list)

    # extractor
    extractor = TextExtractor(model_type=model_type)

    # iterate over test data to parse prediction
    predictions = []
    error_case = []
    for index, row in tqdm(
        enumerate(data),
        total=len(data), 
        desc="Computing Scores"
    ):
        # stop if the file does not exist
        if not Path(output_folder, f"{index}.txt").exists():
            break

        # load response
        with open(Path(output_folder, f"{index}.txt"), "r") as infile:
            response = json.load(infile)

        # parse prediction from the response
        predicted_text = extractor(response)
        prediction = re.search(r"\[(.*?)\]", predicted_text)
        
        if prediction:
            prediction = prediction.group(1).lower()
        else:
            prediction = "[other]"
            error_case.append({
                "index": index,
                "predicted_text": predicted_text,
            })
        predictions.append(prediction)

    # comptue score
    y_true = [sample["label"] for sample in data]
    report = classification_report(
        y_true=y_true,
        y_pred=predictions,
        zero_division=0,
        digits=6,
    )
    print(report)

    # get micro f1 score
    scores = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=predictions,
        average="micro",
    )
    print("Micro F1 Score", scores[2])

    print("Error Cases", error_case)

if __name__ == "__main__":
    app()

