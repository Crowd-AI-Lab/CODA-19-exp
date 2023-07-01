import json
from pathlib import Path
import typer
import re
import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm   
import numpy as np

app = typer.Typer()

mapping = {
    "BACKGROUND": "background",
    "OBJECTIVE": "purpose",
    "METHODS": "method",
    "RESULTS": "finding",
    "CONCLUSIONS": "finding",
}

@app.command()
def process_data(
    input_file: Path = typer.Option(...),
    output_file: Path = typer.Option(...),
    coda19_label: bool = typer.Option(False),
    position_encoding: bool = typer.Option(False),
):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        
    data = []
    paper_id = None
    sent_id = 0
    for line in tqdm(lines, desc="Processing PubMed data", total=len(lines)):
        if line.startswith("###"):
            paper_id = line[3:]
            sent_id = 0
            continue

        # use \t to split label and text
        label, text = line.split("\t")
        label = label.strip()
        text = text.strip()

        # label mapping
        if coda19_label:
            label = mapping[label]

        data.append({
            "paper-id": paper_id,
            "text": text,
            "label": label,
            "sentence-id": sent_id,
        })
        sent_id += 1

    # position encoding    
    if position_encoding:
        table = pd.DataFrame(data)
        num_sentence = table.groupby(["paper-id"]).size().reset_index()
        num_sentence = dict(zip(num_sentence["paper-id"], num_sentence[0]))

        for sample in data:
            paper_id = sample["paper-id"]
            sentence_id = sample["sentence-id"]
            position_ratio = sentence_id / num_sentence[paper_id]
            sample["text"] = f"[POSITION={position_ratio:.2f}] {sample['text']}"

    print(f"There are a total of {len(data)} samples for {input_file}")

    # create folder if not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)


@app.command()
def mix_data(
    input_file_1: Path = typer.Option(...),
    input_file_2: Path = typer.Option(...),
    output_file: Path = typer.Option(...),
    upsampling: bool = typer.Option(False),
    ratio: float = typer.Option(10.0),
):
    # load data
    with open(input_file_1, 'r', encoding='utf-8') as infile:
        data_1 = json.load(infile)

    with open(input_file_2, 'r', encoding='utf-8') as infile:
        data_2 = json.load(infile)

    if upsampling:
        # identify the minority
        num_1 = len(data_1)
        num_2 = len(data_2)
        if num_1 > num_2:
            data = data_2 * int(ratio) + data_1
        else:
            data = data_1 * int(ratio) + data_2
    else:
        data = data_1 + data_2

    # print num of samples
    print(f"There are a total of {len(data_1)} samples for {input_file_1}")
    print(f"There are a total of {len(data_2)} samples for {input_file_2}")
    print(f"There are a total of {len(data)} samples")

    # create folder if not exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2)

@app.command()
def check_length(
    input_file: Path = typer.Option(...),
):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    length = []
    for sample in tqdm(
        data, desc="Counting length", total=len(data)
    ):
        text = sample["text"]
        tokens = word_tokenize(text)
        length.append(len(tokens))
    
    length = np.array(length)
    print(f"Max length: {length.max()}")
    print(f"Min length: {length.min()}")
    print(f"Mean length: {length.mean()}")
    print(f"Median length: {np.median(length)}")

    # print percentiles
    for i in range(80, 101, 5):
        print(f"{i}th percentile: {np.percentile(length, i)}")

if __name__ == "__main__":
    app()