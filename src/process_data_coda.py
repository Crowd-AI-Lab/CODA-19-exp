import os
import json
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def process_coda_data(
    input_folder: Path = typer.Option(..., help="The human_label folder containing the CODA-19 data"),
    output_filename: Path = typer.Option(..., help="The output file name"),
    position_encoding: bool = typer.Option(False, help="Whether to add position encoding"),
):
    # create output folder
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # organize data
    results = []
    max_index = 0
    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"): continue

        filepath = input_folder / filename
        with open(filepath, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        samples = []
        for paragraph in data["abstract"]:
            for sentence in paragraph["sentences"]:
                for segment in sentence:
                    samples.append({
                        "text": segment["segment_text"],
                        "label": segment["crowd_label"],
                    })
        
        # add position_encoding
        if position_encoding:
            total_sentence = len(samples)
            max_index = max(max_index, total_sentence)
            for start_index, sample in enumerate(samples):
                end_index = total_sentence - start_index - 1
                position_ratio = start_index / total_sentence
                sample["text"] = f"[POSITION={position_ratio:.2f}] {sample['text']}"
        
        results.extend(samples)
    
    print(f"There are a total of {len(results)} samples for {input_folder}")
    if position_encoding:
        print(f"The maximum number of sentences in a paragraph is {max_index}")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=2)

if __name__ == "__main__":
    app()