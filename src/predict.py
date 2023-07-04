from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import os
import json
import typer

app = typer.Typer()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
mapping = {
    "BACKGROUND": "background",
    "OBJECTIVE": "purpose",
    "METHODS": "method",
    "RESULTS": "finding",
    "CONCLUSIONS": "finding",
}

@app.command()
@torch.no_grad()
def predict(
    model_name_or_path: str = typer.Option(..., help="Path to pretrained model or model identifier from huggingface.co/models"),
    test_filename: str = typer.Option(..., help="Path to test file"),
    text_key: str = typer.Option("text", help="Key of text in test file"),
    output_filename: str = typer.Option(..., help="Path to output file"),
    batch_size: int = typer.Option(32, help="Batch size"),
    device: str = typer.Option("cuda:0", help="Device to use"),
    label_mapping: bool = typer.Option(False, help="Whether to map label to CODA space"),
):
    # args = parse_args()

    # load model
    print("Loading model...")
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    # load dataset (json file)
    print("Loading dataset...")
    with open(test_filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    print("Number of samples:", len(data))

    # make prediction
    predictions = []
    for start_index in tqdm(range(0, len(data), batch_size)):
        batch = data[start_index: start_index + batch_size]
        batch = tokenizer(
            [sample[text_key] for sample in batch],
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=config.max_position_embeddings,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        # get prediction
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1)
        predictions.extend(prediction.tolist())

    # map prediction to label
    label_dict = config.id2label
    predictions = [label_dict[p] for p in predictions]

    # save prediction to a tsv file
    table = pd.DataFrame({
        "index": np.arange(len(predictions)), 
        "prediction": predictions,
    })

    if label_mapping:
        table["prediction"] = table["prediction"].map(mapping)

    table.to_csv(output_filename, sep="\t", index=False)

if __name__ == "__main__":
    app()